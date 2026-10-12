//! Derived performance metrics: tokens/sec, latency, per-request records and
//! history ring buffers for sparklines. Everything here comes from deltas of
//! the server's own counters between polls, not from guessed constants.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::gpu::GpuStats;
use crate::host::HostSample;
use crate::observe::{LiveStats, SpecMetrics};

pub const HISTORY: usize = 240;
const MAX_REQUESTS: usize = 32;

#[derive(Debug, Clone)]
pub struct RequestRecord {
    pub id_task: i64,
    pub started: Instant,
    pub first_token: Option<Instant>,
    /// Server-measured TTFT in seconds (vLLM histogram). None on
    /// llama.cpp, where the first decoded token's poll is the clock.
    /// Wins over `first_token` wherever a TTFT is shown or divided by.
    pub ttft: Option<f32>,
    /// The request's TOTAL inter-token latency as measured by the server
    /// (vLLM): one ITL sample per decode step — the whole decode span, so
    /// the decode rate is `(decoded - 1) / itl_sum`.
    pub itl_sum: Option<f32>,
    pub ended: Option<Instant>,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub decoded: usize,
    /// Prompt tokens actually pushed through prefill (prompt minus cache hits).
    pub prefill_tokens: usize,
    pub prefill_secs: f32,
    /// Engine-measured prefill mean, preferred to polling estimates.
    pub measured_prefill_tps: Option<f32>,
    /// Engine-derived whole-request decode mean (mlx-serve: the span is
    /// TTFT minus the measured prefill span). Beats the poll-sum decode
    /// rate: mlx's counters move only at completion, so every poll-based
    /// decode rate for it is `whole request / one poll interval`.
    pub measured_decode_tps: Option<f32>,
    pub decode_secs: f32,
    pub peak_decode_tps: f32,
}

impl RequestRecord {
    fn new(id_task: i64, now: Instant, s: &LiveStats) -> Self {
        Self {
            id_task,
            started: now,
            first_token: None,
            ttft: None,
            itl_sum: None,
            ended: None,
            prompt_tokens: s.prompt_tokens,
            cached_tokens: s.cache_tokens,
            decoded: s.decoded,
            prefill_tokens: 0,
            prefill_secs: 0.0,
            measured_prefill_tps: None,
            measured_decode_tps: None,
            decode_secs: 0.0,
            peak_decode_tps: 0.0,
        }
    }

    /// Time to first token. The server-measured TTFT (vLLM) wins:
    /// with that counter set the first token is only observed on the
    /// completion poll, so a first_token - started span would include
    /// the whole decode phase. llama.cpp keeps the poll-estimated one.
    pub fn ttft(&self) -> Option<Duration> {
        if let Some(secs) = self.ttft {
            return Some(Duration::from_secs_f64(secs as f64));
        }
        self.first_token.map(|t| t - self.started)
    }

    pub fn duration(&self, now: Instant) -> Duration {
        self.ended.unwrap_or(now) - self.started
    }

    pub fn avg_decode_tps(&self) -> f32 {
        // Mlx's engine-derived mean wins: the engine published the span
        // (TTFT minus its measured prefill), so both numerators are exact.
        if let Some(rate) = self.measured_decode_tps {
            return rate;
        }
        // The server-measured decode span (vLLM ITL histogram) wins
        // over the poll-interval sum, which only counts the polls that
        // happened to carry a new token.
        if let Some(itl) = self.itl_sum {
            if self.decoded > 1 && itl > 0.0 {
                // itl_sum is the request's TOTAL inter-token time
                // (vLLM observes one sample per decode step); the
                // request produced (decoded - 1) gaps.
                return (self.decoded - 1) as f32 / itl;
            }
        }
        if self.decode_secs > 0.05 {
            self.decoded as f32 / self.decode_secs
        } else {
            0.0
        }
    }

    pub fn avg_prefill_tps(&self) -> f32 {
        if let Some(rate) = self.measured_prefill_tps {
            // Mlx's engine-published mean: exact numerator (the step
            // counter's whole-window movement) and denominator (the
            // window's span), so it wins over prompt-delta / TTFT —
            // mlx's prompt counter credits this request with every
            // request's prefix-cache restores.
            return rate;
        }
        // The server-measured TTFT (vLLM) is the true prefill duration
        // — total prefill tokens divided by TTFT. Without it
        // (llama.cpp) fall back to the poll-interval sum.
        if let Some(ttft) = self.ttft {
            if ttft > 0.0 {
                return self.prefill_tokens as f32 / ttft;
            }
        }
        if self.prefill_secs > 0.05 {
            self.prefill_tokens as f32 / self.prefill_secs
        } else {
            0.0
        }
    }

    pub fn is_live(&self) -> bool {
        self.ended.is_none()
    }

    fn set_prefill_timing(&mut self, tokens: usize, secs: f64) {
        self.prefill_tokens = tokens;
        self.prefill_secs = secs as f32;
        self.measured_prefill_tps = Some(if secs > 0.0 {
            (tokens as f64 / secs) as f32
        } else {
            0.0
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Prefill,
    Decode,
}

/// Token deltas over a sliding time window. Speculative decoders (MTP, draft
/// models) land tokens in bursts, so a per-poll rate flickers; a ~1 s window
/// gives an honest, steady tokens/sec.
#[derive(Debug, Clone)]
pub struct RateWindow {
    span: Duration,
    /// (interval start, interval end, tokens landed in that interval)
    samples: VecDeque<(Instant, Instant, usize)>,
}

impl RateWindow {
    pub fn new(span: Duration) -> Self {
        Self {
            span,
            samples: VecDeque::new(),
        }
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn push(&mut self, from: Instant, to: Instant, tokens: usize) {
        self.samples.push_back((from, to, tokens));
        while let Some((_, end, _)) = self.samples.front() {
            if to.duration_since(*end) > self.span {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Tokens per second over the window. Zero once nothing has landed recently.
    pub fn rate(&self, now: Instant) -> f32 {
        let total: usize = self.samples.iter().map(|(_, _, n)| *n).sum();
        if total == 0 {
            return 0.0;
        }
        let oldest = self.samples.front().map(|(t, _, _)| *t).unwrap_or(now);
        let secs = now.duration_since(oldest).as_secs_f32().max(0.05);
        total as f32 / secs
    }
}

#[derive(Debug)]
pub struct PerfTracker {
    pub started: Instant,
    last_sample: Option<(Instant, LiveStats)>,
    last_gpu: Option<Instant>,
    decode_win: RateWindow,
    prefill_win: RateWindow,
    pub phase: Phase,
    /// Instantaneous (last-interval) rates.
    pub decode_tps: f32,
    pub prefill_tps: f32,
    /// Smoothed rates for the big numerals.
    pub decode_tps_smooth: f32,
    pub prefill_tps_smooth: f32,
    /// The displayed prefill rate is an in-flight reading from the engine's
    /// own step counter (mlx-serve), not the completion-window mean. The
    /// panels tag it "live" while mlx::LiveRates reports it.
    pub prefill_rate_live: bool,
    pub peak_decode_tps: f32,
    pub peak_prefill_tps: f32,
    pub decode_hist: VecDeque<f32>,
    pub prefill_hist: VecDeque<f32>,
    pub util_hist: Vec<VecDeque<f32>>,
    pub power_hist: Vec<VecDeque<f32>>,
    pub temp_hist: Vec<VecDeque<f32>>,
    /// Peak-hold markers for VU meters (fraction 0..1, and time set).
    pub util_peak: Vec<(f32, Instant)>,
    pub power_peak: Vec<(f32, Instant)>,
    pub current: Option<RequestRecord>,
    pub history: VecDeque<RequestRecord>,
    pub session_decoded: u64,
    pub session_prefilled: u64,
    pub session_requests: u64,
    /// Requests pushed to `history` so far; `history` itself is capped.
    pub finished: u64,
    pub total_power_w: f32,
    pub samples: u64,
    pub poll_ok: bool,
    /// Speculative decoding (MTP / draft) — only when `/metrics` is served.
    pub spec: SpecStats,
    /// Memory pipeline meters (`b` view).
    pub bw: BandwidthStats,
}

#[derive(Debug, Clone)]
pub struct SpecStats {
    pub available: bool,
    pub totals: SpecMetrics,
    last: Option<SpecMetrics>,
    draft_win: RateWindow,
    accept_win: RateWindow,
    steps_win: RateWindow,
    last_time: Option<Instant>,
    /// Windowed acceptance rate (0..1), and its history for the sparkline.
    pub accept_rate: f32,
    pub accept_hist: VecDeque<f32>,
    /// Windowed mean accepted draft tokens per verification step.
    pub mean_accepted: f32,
    pub steps_per_sec: f32,
    pub drafts_per_sec: f32,
}

impl SpecStats {
    fn new() -> Self {
        Self {
            available: false,
            totals: SpecMetrics::default(),
            last: None,
            draft_win: RateWindow::new(Duration::from_millis(1500)),
            accept_win: RateWindow::new(Duration::from_millis(1500)),
            steps_win: RateWindow::new(Duration::from_millis(1500)),
            last_time: None,
            accept_rate: 0.0,
            accept_hist: VecDeque::with_capacity(HISTORY),
            mean_accepted: 0.0,
            steps_per_sec: 0.0,
            drafts_per_sec: 0.0,
        }
    }

    /// Session-wide acceptance rate from the cumulative counters.
    pub fn session_accept_rate(&self) -> f32 {
        if self.totals.draft_tokens == 0 {
            0.0
        } else {
            self.totals.accepted as f32 / self.totals.draft_tokens as f32
        }
    }

    /// Fraction of generated tokens that came from accepted drafts.
    pub fn session_draft_share(&self) -> f32 {
        if self.totals.tokens_predicted == 0 {
            0.0
        } else {
            (self.totals.accepted as f32 / self.totals.tokens_predicted as f32).clamp(0.0, 1.0)
        }
    }

    fn observe(&mut self, m: &SpecMetrics, now: Instant) {
        self.available = true;
        if let (Some(prev), Some(t0)) = (&self.last, self.last_time) {
            let d = |a: u64, b: u64| a.saturating_sub(b) as usize;
            let busy = m.busy_secs - prev.busy_secs;
            let t0 = if busy > 0.0 {
                now.checked_sub(Duration::from_secs_f64(busy)).unwrap_or(t0)
            } else {
                t0
            };
            self.draft_win
                .push(t0, now, d(m.draft_tokens, prev.draft_tokens));
            self.accept_win.push(t0, now, d(m.accepted, prev.accepted));
            self.steps_win
                .push(t0, now, d(m.verify_steps, prev.verify_steps));
            let drafts = self.draft_win.rate(now);
            let acc = self.accept_win.rate(now);
            let steps = self.steps_win.rate(now);
            self.drafts_per_sec = drafts;
            self.steps_per_sec = steps;
            // Only while drafting: an idle window keeps the last measured
            // acceptance (shown muted) instead of reading as 0%, and the
            // sparkline stays a real trace.
            if drafts > 0.0 {
                self.accept_rate = (acc / drafts).clamp(0.0, 1.0);
                self.mean_accepted = if steps > 0.0 { acc / steps } else { 0.0 };
                push(&mut self.accept_hist, self.accept_rate);
            }
        }
        self.totals = m.clone();
        self.last = Some(m.clone());
        self.last_time = Some(now);
    }
}

/// One VU-style channel: current value, a held peak that sits then falls,
/// the session maximum for auto-scaling, and a history ring.
#[derive(Debug, Clone)]
pub struct Meter {
    pub value: f32,
    pub hold: f32,
    hold_at: Instant,
    pub max_seen: f32,
    floor: f32,
    /// Fixed full scale (e.g. 100 for a percentage, a PCIe link cap); None = auto.
    pub full_scale: Option<f32>,
    pub hist: VecDeque<f32>,
}

impl Meter {
    pub fn auto(floor: f32) -> Self {
        Self::new(floor, None)
    }

    pub fn fixed(full_scale: f32) -> Self {
        Self::new(full_scale, Some(full_scale))
    }

    fn new(floor: f32, full_scale: Option<f32>) -> Self {
        Self {
            value: 0.0,
            hold: 0.0,
            hold_at: Instant::now(),
            max_seen: 0.0,
            floor,
            full_scale,
            hist: VecDeque::with_capacity(HISTORY),
        }
    }

    pub fn update(&mut self, v: f32, now: Instant, dt: f32) {
        let v = if v.is_finite() { v.max(0.0) } else { 0.0 };
        let v = match self.full_scale {
            Some(fs) => v.min(fs),
            None => v,
        };
        self.value = v;
        self.max_seen = self.max_seen.max(v);
        // Hold 1.2 s, then fall at 40 % of full scale per second.
        if v >= self.hold {
            self.hold = v;
            self.hold_at = now;
        } else if (now - self.hold_at).as_secs_f32() > 1.2 {
            self.hold = (self.hold - 0.4 * self.scale() * dt).max(v);
        }
        push(&mut self.hist, v);
    }

    pub fn scale(&self) -> f32 {
        self.full_scale
            .unwrap_or_else(|| self.max_seen.max(self.floor))
    }

    pub fn frac(&self) -> f32 {
        (self.value / self.scale().max(1e-6)).clamp(0.0, 1.0)
    }

    pub fn hold_frac(&self) -> f32 {
        (self.hold / self.scale().max(1e-6)).clamp(0.0, 1.0)
    }
}

/// Rates along the weight path: disk → RAM → PCIe → VRAM → prefill → decode.
/// Disk and PCIe come from host counters, the RAM and VRAM weight streams
/// are `bytes per step × steps per second` from the GGUF tensor table.
#[derive(Debug, Clone)]
pub struct BandwidthStats {
    last_host: Option<(Instant, HostSample)>,
    pub host: HostSample,
    pub host_seen: bool,
    /// System RAM in use (total minus available), percent, one per host poll.
    pub ram_used_hist: VecDeque<f32>,
    /// System-wide disk reads, MB/s.
    pub disk: Meter,
    /// The inference process's own storage reads, MB/s.
    pub proc_disk_mb_s: f32,
    pub majflt_per_s: f32,
    /// Estimated CPU-side weight stream out of system RAM, GB/s.
    pub ram: Meter,
    /// PCIe host→device per GPU index, MB/s; scale is the link cap when known.
    pub pcie_rx: Vec<Meter>,
    pub pcie_tx: Vec<f32>,
    /// nvidia-smi memory-controller busy %, per GPU index.
    pub vram_busy: Vec<Meter>,
    /// Unified-memory pool fill % (server-reported occupancy over system
    /// RAM), per GPU index. The stage the busy meter cannot cover on a part
    /// without device memory: it always reads 0 there.
    pub vram_pool: Vec<Meter>,
    /// Estimated weight stream out of VRAM across all GPUs, GB/s.
    pub vram: Meter,
    pub prefill: Meter,
    pub decode: Meter,
    /// Latest weight layout the estimates were made from.
    pub layout: WeightLayout,
    /// Weight reads per second: verification steps/s under speculative
    /// decoding, tokens/s otherwise, tokens/ubatch during prefill.
    pub steps_per_s: f32,
}

/// Where the model's bytes sit, from the GGUF tensor table plus VRAM use.
#[derive(Debug, Clone, Default)]
pub struct WeightLayout {
    pub known: bool,
    pub total_bytes: u64,
    /// Bytes touched per weight read (MoE: routed experts only, no embedding).
    pub active_bytes: u64,
    /// Weights that did not fit in VRAM (estimate: file size minus per-GPU
    /// weight share), read from system RAM by the CPU every step.
    pub cpu_bytes: u64,
    /// llama.cpp `--ubatch-size` (weights are read once per micro-batch in prefill).
    pub ubatch: usize,
}

impl WeightLayout {
    pub fn active_frac(&self) -> f32 {
        if self.total_bytes == 0 {
            1.0
        } else {
            (self.active_bytes as f32 / self.total_bytes as f32).clamp(0.0, 1.0)
        }
    }

    /// Bytes streamed per step from RAM and from VRAM respectively.
    pub fn per_step(&self) -> (f32, f32) {
        let cpu_active = self.cpu_bytes as f32 * self.active_frac();
        let vram_active = (self.active_bytes as f32 - cpu_active).max(0.0);
        (cpu_active, vram_active)
    }
}

impl BandwidthStats {
    fn new() -> Self {
        Self {
            last_host: None,
            host: HostSample::default(),
            host_seen: false,
            ram_used_hist: VecDeque::with_capacity(HISTORY),
            disk: Meter::auto(500.0),
            proc_disk_mb_s: 0.0,
            majflt_per_s: 0.0,
            ram: Meter::auto(20.0),
            pcie_rx: Vec::new(),
            pcie_tx: Vec::new(),
            vram_busy: Vec::new(),
            vram_pool: Vec::new(),
            vram: Meter::auto(100.0),
            prefill: Meter::auto(100.0),
            decode: Meter::auto(10.0),
            layout: WeightLayout::default(),
            steps_per_s: 0.0,
        }
    }

    fn ensure_gpu(&mut self, n: usize, gpus: &[GpuStats]) {
        while self.pcie_rx.len() < n {
            self.pcie_rx.push(Meter::auto(1000.0));
            self.pcie_tx.push(0.0);
            self.vram_busy.push(Meter::fixed(100.0));
            self.vram_pool.push(Meter::fixed(100.0));
        }
        for g in gpus {
            let i = g.index as usize;
            if i < self.pcie_rx.len() {
                self.pcie_rx[i].full_scale = pcie_link_mb_s(g.pcie_gen, g.pcie_width);
            }
        }
    }

    fn observe_host(&mut self, s: &HostSample, now: Instant) {
        self.host_seen = true;
        if let Some((t0, prev)) = &self.last_host {
            let dt = (now - *t0).as_secs_f32().max(1e-3);
            let rate = |a: Option<u64>, b: Option<u64>| -> Option<f32> {
                match (a, b) {
                    (Some(a), Some(b)) => Some(a.saturating_sub(b) as f32 / dt),
                    _ => None,
                }
            };
            let disk = rate(s.disk_read_bytes, prev.disk_read_bytes).unwrap_or(0.0) / 1e6;
            self.disk.update(disk, now, dt);
            self.proc_disk_mb_s =
                rate(s.proc_read_bytes, prev.proc_read_bytes).unwrap_or(0.0) / 1e6;
            self.majflt_per_s = rate(s.proc_majflt, prev.proc_majflt).unwrap_or(0.0);
            let n = s
                .pcie_mb_s
                .iter()
                .map(|(i, _, _)| *i as usize + 1)
                .max()
                .unwrap_or(0);
            self.ensure_gpu(n, &[]);
            for (i, rx, tx) in &s.pcie_mb_s {
                let i = *i as usize;
                // dmon occasionally prints a garbage sample (hundreds of GB/s
                // on a gen-3 link); nothing real exceeds the link, so such a
                // sample is dropped rather than shown as a saturated bus.
                let cap = self.pcie_rx[i].full_scale.unwrap_or(PCIE_SANE_MB_S) * 1.05;
                if *rx <= cap {
                    self.pcie_rx[i].update(*rx, now, dt);
                }
                if *tx <= cap {
                    self.pcie_tx[i] = *tx;
                }
            }
        }
        if let (Some(t), Some(a)) = (s.mem_total_bytes, s.mem_available_bytes) {
            if t > 0 {
                push(
                    &mut self.ram_used_hist,
                    t.saturating_sub(a) as f32 / t as f32 * 100.0,
                );
            }
        }
        self.host = s.clone();
        self.last_host = Some((now, s.clone()));
    }
}

/// Above any link that exists today; used to reject bogus samples when the
/// link generation is unknown.
const PCIE_SANE_MB_S: f32 = 130_000.0;

/// PCIe link ceiling in MB/s for a generation and lane count (payload rate
/// after encoding: 0.985 GB/s per lane at gen 3, doubling per generation).
pub fn pcie_link_mb_s(gen: u32, width: u32) -> Option<f32> {
    if gen == 0 || width == 0 {
        return None;
    }
    let per_lane = match gen {
        1 => 250.0,
        2 => 500.0,
        3 => 985.0,
        4 => 1969.0,
        5 => 3938.0,
        _ => 7877.0,
    };
    Some(per_lane * width as f32)
}

impl PerfTracker {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            last_sample: None,
            last_gpu: None,
            decode_win: RateWindow::new(Duration::from_millis(1000)),
            prefill_win: RateWindow::new(Duration::from_millis(600)),
            phase: Phase::Idle,
            decode_tps: 0.0,
            prefill_tps: 0.0,
            decode_tps_smooth: 0.0,
            prefill_tps_smooth: 0.0,
            prefill_rate_live: false,
            peak_decode_tps: 0.0,
            peak_prefill_tps: 0.0,
            decode_hist: VecDeque::with_capacity(HISTORY),
            prefill_hist: VecDeque::with_capacity(HISTORY),
            util_hist: Vec::new(),
            power_hist: Vec::new(),
            temp_hist: Vec::new(),
            util_peak: Vec::new(),
            power_peak: Vec::new(),
            current: None,
            history: VecDeque::with_capacity(MAX_REQUESTS),
            session_decoded: 0,
            session_prefilled: 0,
            session_requests: 0,
            finished: 0,
            total_power_w: 0.0,
            samples: 0,
            poll_ok: false,
            spec: SpecStats::new(),
            bw: BandwidthStats::new(),
        }
    }

    pub fn observe_host(&mut self, s: &HostSample, now: Instant) {
        self.bw.observe_host(s, now);
    }

    /// Re-derive the RAM / VRAM weight streams from the current step rate.
    /// Called once per frame with the latest weight layout.
    pub fn tick_bandwidth(&mut self, layout: &WeightLayout, now: Instant, dt: f32) {
        self.bw.layout = layout.clone();
        let steps = match self.phase {
            Phase::Decode => {
                if self.spec.available && self.spec.steps_per_sec > 0.0 {
                    self.spec.steps_per_sec
                } else {
                    self.decode_tps_smooth
                }
            }
            Phase::Prefill => self.prefill_tps_smooth / layout.ubatch.max(1) as f32,
            Phase::Idle => 0.0,
        };
        self.bw.steps_per_s = steps;
        let (ram_b, vram_b) = layout.per_step();
        let (ram, vram) = if layout.known {
            (ram_b * steps / 1e9, vram_b * steps / 1e9)
        } else {
            (0.0, 0.0)
        };
        self.bw.ram.update(ram, now, dt);
        self.bw.vram.update(vram, now, dt);
    }

    pub fn observe_spec(&mut self, m: &SpecMetrics, now: Instant) {
        self.spec.observe(m, now);
    }

    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }

    /// Tokens generated per joule across all monitored GPUs.
    pub fn tokens_per_joule(&self) -> f32 {
        if self.total_power_w > 1.0 {
            self.decode_tps_smooth / self.total_power_w
        } else {
            0.0
        }
    }

    pub fn observe(&mut self, s: &LiveStats, now: Instant) {
        self.samples += 1;
        self.poll_ok = true;
        let Some((t0, prev)) = self.last_sample.clone() else {
            self.last_sample = Some((now, s.clone()));
            if s.processing {
                self.current = Some(RequestRecord::new(s.id_task, now, s));
                self.session_requests += 1;
            }
            return;
        };
        let dt = (now - t0).as_secs_f32().max(1e-3);

        let new_request = s.id_task != prev.id_task
            || (s.processing && !prev.processing)
            || (s.processing && s.decoded < prev.decoded);
        if new_request && s.processing {
            if let Some(mut done) = self.current.take() {
                // A record with no spans of its own (mlx: the completion's
                // counters moved on a scrape between polls, so the record
                // never saw its tokens move) is filled entirely by the
                // closing view: the engine's exact spans replace the poll
                // arithmetic that would otherwise divide the request's whole
                // token count by one poll interval. A record that *did*
                // observe its own traffic (strata's live reads) keeps those
                // counts; the closing view only tops up rates there.
                if let Some(close) = s.closing.as_ref() {
                    // The closing view's spans come from the adapter's
                    // whole-counter window, which a busy server inflates
                    // with other requests' prefix-cache restores; its
                    // *exact pair* — the per-request histograms the engine
                    // credits at completion — names the finished request
                    // alone. So the pair is the final word on the spans:
                    // whether or not the record watched any of its own
                    // traffic (mlx's counters freeze until completion, so
                    // it usually watches none), the row reads through the
                    // exact counts and the session tally follows the
                    // substitution. Strata publishes no exact pair and is
                    // served by the `prefill_secs` path below.
                    // Two exact ledgers, one precedence: the whole-token
                    // pair (`published_prompt`) is the adapter's own two-
                    // scrape delta — mlx credits a request's whole prompt
                    // there in one block at its completion and nothing
                    // foreign rides that ledger (the prefix-cache flood
                    // moves `cached_tokens_total`, a different counter).
                    // The per-request histogram pair (`exact_prompt`) is
                    // finer but shared: foreign completions move it too,
                    // and the adapter has already refused it — reads 0 —
                    // when a window was dishonest. Whichever ledger names
                    // the finished request sets the spans.
                    // The whole-counter window is only a single request's
                    // ledger when exactly one completion moved it. A close
                    // that absorbed no completion (strata's: its engine's
                    // completion count moved across someone else's window)
                    // credits nothing, and a window whose span outruns the
                    // request's published size absorbed other requests'
                    // publications (the flood's 12,070 where 9,698
                    // belonged — the churner's spans land between the
                    // adapter's scrapes). A *smaller* window with one
                    // completion is chunked prefill: the engine published
                    // all of it, so the exact ledgers read through.
                    let one_completion = close.published_completions == 1;
                    // Strata names no published pair and its engine gives
                    // no completion count (both read 0); its close carries
                    // the exact `prefill_secs` span which is honest
                    // whatever the window watched, so a strata close names
                    // its request through the span path.
                    let honest_close = one_completion || close.published_completions == 0;
                    // The published-span ceiling applies to the closing
                    // view too: a close built across the adapter's blind
                    // scrape carries the flood's movement in its spans.
                    // (Its measured rates are its own — a live-rate model
                    // never divided the flood by a span it did not publish.)
                    let close_prompt =
                        if close.span_known > 0 && close.prompt > close.span_known as usize {
                            close.span_known as usize
                        } else {
                            close.prompt
                        };
                    // The exact pair names the finished request only when
                    // the close moved the per-request histograms by exactly
                    // one completion; the published pair (the adapter's own
                    // two-scrape bracket) names it across the same window.
                    // Whichever ledger speaks sets the spans, and every
                    // span the record ends with passes the published-span
                    // ceiling: the engine named this request's prompt
                    // whole, an adapter view that outruns it divided the
                    // flood, not the request.
                    let named = if close.exact_prompt > 0 {
                        close.exact_prompt.min(close_prompt as u64)
                    } else if close.published_prompt > 0 && one_completion {
                        close.published_prompt.min(close_prompt as u64)
                    } else {
                        0
                    };
                    let named_gen = if close.exact_gen > 0 {
                        close.exact_gen
                    } else {
                        close.published_gen
                    };
                    let named_pre = named.saturating_sub(close.cached as u64);
                    if named > 0 {
                        // An exact ledger names the finished request's
                        // sizes: the row reads through it, whatever the
                        // record watched through the whole-counter window.
                        if done.prompt_tokens != named as usize {
                            self.session_prefilled = self
                                .session_prefilled
                                .saturating_sub(done.prefill_tokens as u64)
                                .saturating_add(named_pre);
                            done.prompt_tokens = named as usize;
                            done.cached_tokens = close.cached;
                            done.prefill_tokens = named_pre as usize;
                        }
                        if named_gen > 0 && done.decoded != named_gen as usize {
                            self.session_decoded = self
                                .session_decoded
                                .saturating_sub(done.decoded as u64)
                                .saturating_add(named_gen);
                            done.decoded = named_gen as usize;
                        }
                    } else if done.prefill_tokens <= 1 && done.decoded == 0 {
                        // No ledger named the request (plain vLLM publishes
                        // none; on mlx the adapter refused them — a batch
                        // of completions, or a publication between scrapes)
                        // and the record watched none of its own traffic:
                        // the closing view's spans, under the ceiling, are
                        // the record's whole request — the shape this
                        // rescue was written for.
                        done.prompt_tokens = close_prompt;
                        done.cached_tokens = close.cached.min(close_prompt);
                        done.prefill_tokens = close_prompt.saturating_sub(close.cached);
                        done.decoded = close.gen;
                        if close.ttft_secs > 0.0 {
                            done.ttft = Some(close.ttft_secs as f32);
                        }
                        if close.itl_sum > 0.0 {
                            done.itl_sum = Some(close.itl_sum as f32);
                        }
                        // The closing poll re-anchored the baselines, so
                        // its deltas were 0 — bill the rescued counters to
                        // the session totals explicitly.
                        self.session_prefilled += done.prefill_tokens as u64;
                        self.session_decoded += done.decoded as u64;
                    } else if close_prompt > done.prompt_tokens && close_prompt > s.prompt_processed
                    {
                        // The published-span fallback: neither ledger named
                        // the request, but the close's window outruns the
                        // span the record's request published — the excess
                        // is other requests' publications the adapter's
                        // blind scrape absorbed. Read the row back through
                        // the published size.
                        done.prompt_tokens = s.prompt_processed;
                        done.cached_tokens = done.cached_tokens.min(done.prompt_tokens);
                        done.prefill_tokens = done.prompt_tokens - done.cached_tokens;
                        if close.gen > done.decoded {
                            self.session_decoded += (close.gen - done.decoded) as u64;
                            done.decoded = close.gen;
                        }
                        self.session_prefilled += done.prefill_tokens as u64;
                    }
                    // Engine-measured spans ride the close whatever the
                    // ledger that named the sizes. Identity, the same
                    // rule the in-loop stamps use: a close built on the
                    // busy completion edge names the successor's task (the
                    // finished request keeps its record's id), and the
                    // finished row's rescue still owns this close's rates —
                    // the adapter staged them for the request it closed.
                    if honest_close {
                        // Strata's exact span, whole spans, and the
                        // engine's measured rates ride the finished
                        // record whatever ledger set its sizes: the close
                        // is built from the engine's own record of the
                        // *finished* request, so its spans are exact and
                        // the window's readings were estimates. The
                        // session tally follows the substitution.
                        if let Some(secs) = close.prefill_secs {
                            let tokens = close_prompt.saturating_sub(close.cached);
                            self.session_prefilled = self
                                .session_prefilled
                                .saturating_sub(done.prefill_tokens as u64)
                                .saturating_add(tokens as u64);
                            self.session_decoded += close.gen.saturating_sub(done.decoded) as u64;
                            done.prompt_tokens = close_prompt;
                            done.cached_tokens = close.cached.min(close_prompt);
                            done.decoded = done.decoded.max(close.gen);
                            done.set_prefill_timing(tokens, secs);
                        }
                        if let Some(rate) = close.measured_prefill_tps {
                            done.measured_prefill_tps = Some(rate);
                        }
                        if let Some(rate) = close.measured_decode_tps {
                            done.measured_decode_tps = Some(rate);
                        }
                    }
                    if let Some(rate) = close.measured_prefill_tps {
                        done.measured_prefill_tps = Some(rate);
                    }
                    if let Some(rate) = close.measured_decode_tps {
                        done.measured_decode_tps = Some(rate);
                    }
                    // TTFT / ITL: the in-loop exact-pair stamps already
                    // gave the finishing record the engine's histogram
                    // readings on the completion scrape (`ttft_secs` is
                    // that histogram's mean, not a per-request span; the
                    // idle scrape's close carries it exact for engines
                    // that move their ledgers when the phase goes dark).
                    // The rescue must not re-stamp a busy scrape's mean
                    // over the honest span the record already holds.
                    if close.ttft_secs > 0.0
                        && (close.id_task == done.id_task
                            || (done.ttft.is_none() && done.prefill_tokens <= 1))
                        && (done.ttft.is_none()
                            || done
                                .ttft
                                .is_some_and(|t| (t as f64) < close.ttft_secs * 0.5))
                    {
                        done.ttft = Some(close.ttft_secs as f32);
                    }
                    if close.itl_sum > 0.0 {
                        done.itl_sum = Some(close.itl_sum as f32);
                    }
                    // Physical ceiling: P prompt tokens in T seconds of
                    // measured TTFT cannot beat P/T — the row's printed
                    // rate reads back through the span it finally took.
                    if let (Some(ttft), true) = (done.ttft, done.prompt_tokens > 1) {
                        if ttft > 0.0 {
                            let ceiling = done.prompt_tokens as f32 / ttft;
                            if let Some(rate) = done.measured_prefill_tps {
                                if rate > ceiling {
                                    // An engine-measured rate is not a
                                    // fabrication: a span estimate that
                                    // outruns it divided a window the
                                    // engine never published. Keep the
                                    // engine's own mean.
                                    if done.prefill_secs == 0.0 {
                                        done.measured_prefill_tps = Some(ceiling);
                                    }
                                }
                            }
                        }
                    }
                }
                // Unconditional: only the corrections above are gated on
                // a closing view — a record without one must still stop
                // being "live" or its row pulses with a growing duration
                // forever.
                done.ended = Some(now);
                // Same backdate the closing view's shape gets: the
                // engine's TTFT + total inter-token time is the request's
                // true wall span (admission -> last token); llama.cpp rows
                // have no histograms and keep the detection-based span.
                if done.ttft.is_some_and(|t| t > 0.0) {
                    let e2e = done.ttft.unwrap() as f64 + done.itl_sum.unwrap_or(0.0) as f64;
                    if let Some(st) = now.checked_sub(Duration::from_secs_f64(e2e)) {
                        done.started = st;
                    }
                }
                self.finish(done);
            }
            self.current = Some(RequestRecord::new(s.id_task, now, s));
            self.session_requests += 1;
            self.decode_win.clear();
            self.prefill_win.clear();
        }

        // Deltas only make sense within one request.
        let same = !new_request;
        let d_dec = if same {
            s.decoded.saturating_sub(prev.decoded)
        } else {
            s.decoded
        };
        let d_pre = if same {
            s.prompt_processed.saturating_sub(prev.prompt_processed)
        } else {
            s.prompt_processed
        };
        // Mlx credits a request's whole token span at its *completion*,
        // while the whole-counter window can move while the *successor* is
        // already lit. The vllm-shaped completion edge is the adapter's
        // own: it names the finished request through the per-request
        // histograms (`exact_prompt`) while `closing` rides the same
        // scrape. On that scrape the session tallies' deltas below belong
        // to the finished request — the closing rescue bills them, so
        // counting them here too would double-bill. (Strata publishes
        // neither field on its live scrapes, and strata's own completion
        // rides an idle scrape where this never fires.)
        let completion_edge = s.exact_prompt > 0 && s.closing.is_some();
        // Continuous batching: the adapter's whole-counter window can move
        // while a *different* request is in flight — mlx credits the
        // finished request's whole sizes as a successor keeps `running`
        // lit, and the in-flight record sees the movement it did not earn
        // (the flood's 11,220-token row). The adapter's own attribution is
        // the honest ledger: a lit scrape's whole-counter movement names a
        // completed request (`published_prompt > 0`), so the record-level
        // spans and poll-window rates read blank there; the published pair
        // and the closing rescue own those tokens instead.
        let foreign_move = s.published_prompt > 0 && s.processing;
        let (d_pre_rec, d_dec_rec) = if foreign_move { (0, 0) } else { (d_pre, d_dec) };
        let (d_pre_bill, d_dec_bill) = if completion_edge {
            (0, 0)
        } else {
            (d_pre, d_dec)
        };

        self.phase = if !s.processing {
            Phase::Idle
        } else if d_dec_rec > 0 || (s.decoded > 0 && d_pre_rec == 0) {
            Phase::Decode
        } else {
            Phase::Prefill
        };

        // Windowed rates: steady under bursty speculative decoding, and they
        // fall to zero within a second of the counter stopping.
        self.decode_win.push(t0, now, d_dec_rec);
        self.prefill_win.push(t0, now, d_pre_rec);
        if !s.processing {
            self.decode_win.clear();
            self.prefill_win.clear();
        }
        self.decode_tps = self.decode_win.rate(now);
        self.prefill_tps = self.prefill_win.rate(now);
        // vLLM reports token deltas only at completion, so the windowed
        // prefill rate would divide the whole prompt by one poll
        // interval. When a request closes in this window, use its
        // measured TTFT as the denominator instead (tokens / TTFT).
        if !s.processing && s.ttft_secs > 0.0 && d_pre_rec > 0 {
            self.prefill_tps = self.prefill_tps.max(d_pre_rec as f32 / s.ttft_secs as f32);
        }
        if let Some(rate) = s.prefill_tps {
            self.prefill_tps = rate;
        }
        // Same for decode: vLLM's generation counter moves only on the
        // completion poll — which is exactly when the window above was
        // cleared — so use the measured ITL sum (one sample per decode
        // step, i.e. decoded - 1 of them).
        if !s.processing && s.itl_sum > 0.0 && d_dec_rec > 1 {
            self.decode_tps = self
                .decode_tps
                .max((d_dec_rec - 1) as f32 / s.itl_sum as f32);
        }
        let ema = |prev: f32, x: f32, up: f32, down: f32| -> f32 {
            let a = if x > prev { up } else { down };
            prev + (x - prev) * a
        };
        self.decode_tps_smooth = ema(self.decode_tps_smooth, self.decode_tps, 0.5, 0.25);
        self.prefill_tps_smooth = ema(self.prefill_tps_smooth, self.prefill_tps, 0.5, 0.25);
        self.prefill_rate_live = s.processing && s.prefill_rate_live;
        if self.decode_tps_smooth < 0.05 {
            self.decode_tps_smooth = 0.0;
        }
        if self.prefill_tps_smooth < 0.5 {
            self.prefill_tps_smooth = 0.0;
        }
        self.peak_decode_tps = self.peak_decode_tps.max(self.decode_tps);
        self.peak_prefill_tps = self.peak_prefill_tps.max(self.prefill_tps);
        self.bw.decode.update(self.decode_tps, now, dt);
        self.bw.prefill.update(self.prefill_tps, now, dt);
        push(&mut self.decode_hist, self.decode_tps);
        push(&mut self.prefill_hist, self.prefill_tps);
        self.session_decoded += d_dec_bill as u64;
        self.session_prefilled += d_pre_bill as u64;

        // Two exact ledgers on every vllm-shaped scrape: the per-request
        // histograms (`exact_prompt`, shared — foreign completions move them
        // and the adapter refuses the pair when a window was dishonest) and
        // the whole-token counters' one-scrape delta (`published_prompt`,
        // mlx's own ledger: a request's whole span lands there in a block
        // at its completion, prefix-cache restores included, nothing
        // foreign). Whichever names the finished request sets the spans.
        let exact_prompt = if s.exact_prompt > 0 {
            s.exact_prompt
        } else {
            s.published_prompt
        };
        let exact_gen = if s.exact_gen > 0 {
            s.exact_gen
        } else {
            s.published_gen
        };
        if let Some(cur) = self.current.as_mut() {
            // The slot can still carry the previous request's prompt size on
            // the first sample, so always take the latest value.
            cur.prompt_tokens = s.prompt_tokens;
            cur.cached_tokens = s.cache_tokens;
            cur.decoded = cur.decoded.max(s.decoded);
            if let Some(close) = s.closing.as_ref() {
                // The completion scrape re-anchored the adapter's baselines,
                // so the record's own prompt/cached reads came back blank;
                // the closing view still names them. Take them once — the
                // rescue re-reads the same closing view on the next scrape
                // and must not re-bill the spans it already applied.
                if cur.prompt_tokens == 0 && close.prompt > 0 {
                    cur.prompt_tokens = close.prompt;
                    cur.cached_tokens = close.cached;
                }
            }
            // A lit scrape that moved the whole-token counters completed
            // someone's request — under continuous batching the successor's
            // record watches that happen and must not bill the movement to
            // its own prefill. (The close's rescue reads the same pair and
            // gives it to the finished row; mlx's in-flight request moves
            // `prompt_processed` through its own live span, which is not
            // this ledger.) Strata and llama publish no pair at all.
            if d_pre_rec > 0 {
                cur.prefill_tokens += d_pre_rec;
                cur.prefill_secs += dt;
                // The published-span ceiling, applied after the
                // accumulation and only where the close names *this*
                // record: a completion publishes someone's whole span
                // between two adapter scrapes while a successor is lit,
                // and the closing rescue gives those tokens to the finished
                // row. Where the close does name this record, its prompt
                // clamps to the span the engine published (0 = none).
                let named_here = s
                    .closing
                    .as_ref()
                    .filter(|c| c.id_task == cur.id_task)
                    .map_or(0, |c| c.span_known);
                if named_here > 0 && cur.prompt_tokens > named_here as usize {
                    cur.prompt_tokens = named_here as usize;
                    cur.cached_tokens = cur.cached_tokens.min(cur.prompt_tokens);
                    cur.prefill_tokens = cur.prompt_tokens - cur.cached_tokens;
                }
            }
            if self.phase == Phase::Prefill {
                if let Some(rate) = s.prefill_tps {
                    cur.measured_prefill_tps = Some(rate);
                }
            }
            if let Some(secs) = s.prefill_secs {
                // Strata's completion carries the request's exact prompt
                // size and span: replace the record's live estimates with
                // the engine's final counts. Mlx stamps *its* engine span
                // on the scrape that moves the counters, which under busy
                // batching still shows a successor in flight — that span
                // belongs to the finished request alone, and the adapter
                // has already staged it into the closing view. Claim it
                // only once this record's own request is over.
                if !s.processing {
                    self.session_prefilled = self
                        .session_prefilled
                        .saturating_sub(cur.prefill_tokens as u64)
                        .saturating_add(s.prompt_processed as u64);
                    cur.set_prefill_timing(s.prompt_processed, secs);
                }
            }
            if d_dec_rec > 0 {
                cur.decode_secs += dt;
                cur.peak_decode_tps = cur.peak_decode_tps.max(self.decode_tps);
                if cur.first_token.is_none() {
                    cur.first_token = Some(now);
                }
            }
            // Server-measured timings (vLLM): the poll that closed a
            // request carries its whole TTFT / ITL lifetime in one
            // delta; stamp it onto the record.
            if s.ttft_secs > 0.0 {
                // vLLM's measured TTFT arrives whole on the *completion*
                // scrape — and under continuous batching the completion
                // and a successor's admission share one scrape, so this
                // stats carries the finished request's TTFT while `cur`
                // is already the successor's record. Stamp it only on the
                // record it was measured for; the closing rescue gives it
                // to the finished row, and the successor's own record gets
                // its TTFT on *its* completion scrape.
                if !s.processing {
                    // Only an idle scrape's TTFT can name *this* record's
                    // request: mlx credits the TTFT sums at completion, and
                    // under continuous batching the completion scrape is
                    // already the successor's first sample. vLLM's honest
                    // shape closes on the same scrape the TTFT arrives and
                    // reads this way too (the rescue owns the finished row).
                    cur.ttft = Some(s.ttft_secs as f32);
                } else if s.id_task == cur.id_task {
                    cur.ttft = Some(s.ttft_secs as f32);
                }
            }
            if s.itl_sum > 0.0 {
                cur.itl_sum = Some(s.itl_sum as f32);
            }
            // Precision correction: a record may carry a TTFT stamped from
            // the poll cadence (the window's mean over a scrape that could
            // not see the true span — 0.10s against a real 6.8s). Once this
            // record's own request completes and the engine's TTFT
            // histogram moves by exactly one count, that mean is the
            // request's exact span; read the row back through it. (Only
            // for records that had no honest span: vLLM's completion-edge
            // TTFT arrives exact and never needs correcting.)
            if exact_prompt > 0
                && exact_gen > 0
                && s.ttft_secs > 0.0
                && !s.processing
                && cur.ttft.is_some_and(|t| (t as f64) < s.ttft_secs * 0.5)
            {
                cur.ttft = Some(s.ttft_secs as f32);
            }
            if s.id_task == cur.id_task || !s.processing {
                // Mlx's engine-published mean: exact numerator (the step
                // counter's whole-window movement) and denominator (the
                // window's span), so it wins over prompt-delta / TTFT —
                // mlx's prompt counter credits this request with every
                // request's prefix-cache restores. Attribution: a *lit*
                // scrape's pair describes the request its window says
                // completed — on a busy server that is whoever finished,
                // not whoever the panels opened a row for. Mlx's live meter
                // republishes the in-flight request's rate every scrape,
                // so refusing a foreign pair loses nothing.
                if let Some(rate) = s.measured_prefill_tps {
                    cur.measured_prefill_tps = Some(rate);
                }
                if let Some(rate) = s.measured_decode_tps {
                    // The engine-derived decode mean (TTFT minus the measured
                    // prefill span): mlx's counters freeze until completion,
                    // so the poll-sum rate below reads the whole request over
                    // one poll interval.
                    cur.measured_decode_tps = Some(rate);
                }
            }
            if let Some(secs) = s.prefill_secs {
                // The engine's measured prefill span, stamped when the
                // record has no span of its own (mlx's counters move once,
                // at completion, and the completion scrape can be the
                // first one to carry them). `set_prefill_timing` recomputes
                // the exact rate from the record's own token count.
                if cur.prefill_secs == 0.0 && s.id_task == cur.id_task {
                    // Task identity: a span riding a completion scrape was
                    // measured for the request that *finished*; a
                    // successor's record must not time its own prefill at
                    // the previous request's span. (Mlx's live gauge can
                    // publish a finishing request's `prefill_secs` while
                    // the successor is already lit — the same straddle the
                    // exact pair and TTFT stamps guard.)
                    cur.set_prefill_timing(cur.prefill_tokens, secs);
                }
                // A record whose TTFT the completion counters never
                // named (the completion's scrape carried no TTFT delta:
                // mlx moves its sums while `running` stays lit and the
                // adapter's window can miss them) still knows its prefill
                // span. The engine measured it for this request: it is the
                // one span the flood cannot reach, and it beats printing
                // `0.00s` for a request that clearly ran.
                if cur.ttft.is_none()
                    && secs > 0.0
                    && s.id_task == cur.id_task
                    && (cur.measured_prefill_tps.is_some() || s.exact_prompt > 0)
                {
                    // The span is this record's own (the engine measured a
                    // rate for it, or its exact prompt is known): a request
                    // that clearly ran cannot print `0.00s`. A bare staged
                    // span with no rate and no exact pair is somebody
                    // else's residual note.
                    cur.ttft = Some(secs as f32);
                }
            }
            if exact_prompt > 0 && s.itl_sum > 0.0 {
                // The engine's per-request histogram names the in-flight
                // request's exact prompt size, and it moves exactly once
                // the request completes. A record carrying a bigger span
                // is reading a window that straddled a busy scrape — the
                // same correction the closing rescue makes, applied while
                // the record is still live so a completion that never
                // forms a closing view (the phase-dark scrape's own stats
                // finishing the record) still reads through the exact
                // counts. (Mlx's shape: its ITL histogram exists but moves
                // with completions, and strata publishes no such pair at
                // all — a scrape carrying neither keeps its own exact
                // counters.)
                if cur.prompt_tokens != exact_prompt as usize {
                    let pre = exact_prompt.saturating_sub(cur.cached_tokens as u64);
                    self.session_prefilled = self
                        .session_prefilled
                        .saturating_sub(cur.prefill_tokens as u64)
                        .saturating_add(pre);
                    cur.prompt_tokens = exact_prompt as usize;
                    cur.prefill_tokens = pre as usize;
                }
                if exact_gen > 0 && cur.decoded != exact_gen as usize {
                    self.session_decoded = self
                        .session_decoded
                        .saturating_sub(cur.decoded as u64)
                        .saturating_add(exact_gen);
                    cur.decoded = exact_gen as usize;
                }
            }
            // Physical ceiling on whatever rate the record carries: P
            // prompt tokens in T seconds of measured TTFT cannot beat
            // P/T. Applies on every scrape — the completion's stamped
            // stats go stale under continuous batching (the next poll
            // already belongs to the successor), and a row that printed
            // the flood's rate before its close must read back through
            // the exact counts the moment they land.
            if let (Some(ttft), true) = (cur.ttft, exact_prompt > 0) {
                if ttft > 0.0 {
                    let ceiling = exact_prompt as f32 / ttft;
                    if let Some(rate) = cur.measured_prefill_tps {
                        if rate > ceiling {
                            cur.measured_prefill_tps = Some(ceiling);
                        }
                    }
                }
            }
            if !s.processing {
                let mut done = self.current.take().unwrap();
                done.ended = Some(now);
                // Same backdate as the closing path: vLLM's TTFT +
                // total inter-token time is the request's true wall
                // span (admission -> last token). llama.cpp rows have
                // no histograms and keep the detection-based span.
                if done.ttft.is_some_and(|t| t > 0.0) {
                    let e2e = done.ttft.unwrap() as f64 + done.itl_sum.unwrap_or(0.0) as f64;
                    if let Some(st) = now.checked_sub(Duration::from_secs_f64(e2e)) {
                        done.started = st;
                    }
                }
                self.finish(done);
            }
        }

        self.last_sample = Some((now, s.clone()));
    }

    fn finish(&mut self, mut r: RequestRecord) {
        // The engine's spans beat detection timing when the request's own
        // counters moved between two scrapes (mlx's completion): the
        // in-loop rates divided a whole span by the one interval the poll
        // happened to observe. Decode: TTFT minus the engine's measured
        // prefill span is the request's whole decode lifetime — the same
        // split the closing rescue makes, applied to records that did see
        // their own prefill stream. vLLM's rows keep their ITL histogram.
        if r.measured_decode_tps.is_none()
            && r.itl_sum.is_none()
            && r.prefill_secs > 0.0
            && r.decoded > 1
        {
            let ttft = r.ttft.map_or(0.0, |t| t as f64);
            let decode_secs = ttft - r.prefill_secs as f64;
            if decode_secs > 0.05 {
                r.measured_decode_tps = Some(((r.decoded - 1) as f64 / decode_secs) as f32);
            }
        }
        if r.decoded == 0 && r.prefill_tokens == 0 {
            return;
        }
        if self.history.len() >= MAX_REQUESTS {
            self.history.pop_front();
        }
        self.history.push_back(r);
        self.finished += 1;
    }

    pub fn observe_gpu(&mut self, gpus: &[GpuStats], now: Instant) {
        let dt = self
            .last_gpu
            .map(|t| (now - t).as_secs_f32())
            .unwrap_or(0.2)
            .clamp(0.0, 2.0);
        self.last_gpu = Some(now);
        let n = gpus.iter().map(|g| g.index as usize + 1).max().unwrap_or(0);
        while self.util_hist.len() < n {
            self.util_hist.push(VecDeque::with_capacity(HISTORY));
            self.power_hist.push(VecDeque::with_capacity(HISTORY));
            self.temp_hist.push(VecDeque::with_capacity(HISTORY));
            self.util_peak.push((0.0, now));
            self.power_peak.push((0.0, now));
        }
        self.total_power_w = gpus.iter().map(|g| g.power_watts).sum();
        self.bw.ensure_gpu(n, gpus);
        for g in gpus {
            let i = g.index as usize;
            self.bw.vram_busy[i].update(g.utilization_mem, now, dt);
            // A unified part has no memory controller to watch; its VRAM
            // stage shows how full the shared pool is instead. Occupancy is
            // the server's own report (weight + KV + graph) over system RAM.
            if g.unified && g.mem_total_mb > 0 {
                let fill = g.mem_used_mb as f32 * 100.0 / g.mem_total_mb as f32;
                self.bw.vram_pool[i].update(fill.min(100.0), now, dt);
            }
            push(&mut self.util_hist[i], g.utilization_gpu);
            push(&mut self.power_hist[i], g.power_watts);
            push(&mut self.temp_hist[i], g.temperature.unwrap_or(0.0));
            hold_peak(&mut self.util_peak[i], g.utilization_gpu / 100.0, now, dt);
            hold_peak(&mut self.power_peak[i], g.power_frac(), now, dt);
        }
    }

    /// Rows for the request table, newest first, live request on top.
    pub fn recent_requests(&self, n: usize) -> Vec<&RequestRecord> {
        let mut out: Vec<&RequestRecord> = Vec::with_capacity(n);
        if let Some(c) = &self.current {
            out.push(c);
        }
        out.extend(self.history.iter().rev().take(n.saturating_sub(out.len())));
        out
    }
}

/// VU-style peak hold: sits for 1.2 s then falls at 0.4/s.
fn hold_peak(peak: &mut (f32, Instant), value: f32, now: Instant, dt: f32) {
    if value >= peak.0 {
        *peak = (value, now);
    } else if (now - peak.1).as_secs_f32() > 1.2 {
        peak.0 = (peak.0 - 0.4 * dt).max(value);
    }
}

fn push(h: &mut VecDeque<f32>, v: f32) {
    if h.len() >= HISTORY {
        h.pop_front();
    }
    h.push_back(v);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(
        id: i64,
        processing: bool,
        prompt: usize,
        processed: usize,
        decoded: usize,
    ) -> LiveStats {
        LiveStats {
            ctx_max: 4096,
            prompt_tokens: prompt,
            prompt_processed: processed,
            decoded,
            processing,
            id_task: id,
            ..Default::default()
        }
    }

    #[test]
    fn derives_rates_and_request_record() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, false, 0, 0, 0), t0);
        p.observe(&slot(2, true, 1000, 200, 0), t0 + step);
        assert_eq!(p.phase, Phase::Prefill);
        p.observe(&slot(2, true, 1000, 600, 0), t0 + step * 2);
        assert!(
            (p.prefill_tps - 1500.0).abs() < 1.0,
            "prefill {}",
            p.prefill_tps
        );
        p.observe(&slot(2, true, 1000, 1000, 0), t0 + step * 3);
        p.observe(&slot(2, true, 1000, 1000, 10), t0 + step * 4);
        assert_eq!(p.phase, Phase::Decode);
        p.observe(&slot(2, true, 1000, 1000, 20), t0 + step * 5);
        // 20 tokens landed over the last second of samples.
        assert!((p.decode_tps - 20.0).abs() < 0.5, "decode {}", p.decode_tps);
        p.observe(&slot(2, false, 1000, 0, 20), t0 + step * 6);
        assert_eq!(p.phase, Phase::Idle);
        assert!(p.current.is_none());
        let r = p.history.back().expect("finished request");
        assert_eq!(r.decoded, 20);
        assert_eq!(r.prefill_tokens, 1000);
        assert!(r.ttft().is_some());
        assert!(r.avg_decode_tps() > 40.0);
        assert_eq!(p.session_decoded, 20);
        assert_eq!(p.session_requests, 1);
        // llama.cpp exposes no server timings: the poll-based rates
        // remain the source of truth.
        assert!(r.ttft.is_none());
        assert!(r.itl_sum.is_none());
        assert!(r.measured_prefill_tps.is_none());
        assert!((r.avg_prefill_tps() - 1000.0 / 0.6).abs() < 0.01);
    }

    /// vLLM-shaped slot: counters jump only at completion; the
    /// optional `closing` view is what the adapter hands over when a
    /// completion and a successor's admission share one poll.
    fn vllm_slot(
        id: i64,
        processing: bool,
        prompt: usize,
        processed: usize,
        decoded: usize,
        ttft: f64,
        itl: f64,
        closing: Option<crate::observe::ClosingRequest>,
    ) -> LiveStats {
        LiveStats {
            ctx_max: 4096,
            prompt_tokens: prompt,
            prompt_processed: processed,
            decoded,
            processing,
            id_task: id,
            ttft_secs: ttft,
            itl_sum: itl,
            closing,
            ..Default::default()
        }
    }

    #[test]
    fn vllm_completion_reports_server_measured_rates() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, false, 0, 0, 0), t0);
        // Admission: vLLM moves none of its counters while the request
        // is in flight.
        p.observe(&vllm_slot(2, true, 0, 0, 0, 0.0, 0.0, None), t0 + step);
        assert!(p.prefill_tps < 1.0, "{}", p.prefill_tps);
        // Completion 17 s later: 1000 prompt tokens, 40 generated,
        // 2.0 s TTFT, 0.05 s total inter-token time.
        p.observe(
            &vllm_slot(2, false, 1000, 1000, 40, 2.0, 0.05, None),
            t0 + step + Duration::from_secs(17),
        );
        let r = p.history.back().expect("finished request");
        assert_eq!(r.prefill_tokens, 1000);
        assert!(r.measured_prefill_tps.is_none());
        // The user's formula: total prefill tokens / TTFT.
        assert!(
            (r.avg_prefill_tps() - 500.0).abs() < 1.0,
            "{}",
            r.avg_prefill_tps()
        );
        assert!((r.ttft().map(|d| d.as_secs_f64()).unwrap() - 2.0).abs() < 1e-6);
        // Decode: 39 inter-token gaps over the server-measured
        // span (itl_sum = 0.05 s total).
        assert!(
            (r.avg_decode_tps() - 39.0 / 0.05).abs() < 1.0,
            "{}",
            r.avg_decode_tps()
        );
        // The graph spike is bounded by the same measurement, not by
        // one poll interval (which would say 5000 tok/s here).
        assert!(
            (p.peak_prefill_tps - 500.0).abs() < 1.0,
            "{}",
            p.peak_prefill_tps
        );
        // Decode likewise: the completion poll is the only one where the
        // counter moves, so without the ITL clamp this would read 0.
        assert!(
            (p.peak_decode_tps - 39.0 / 0.05).abs() < 1.0,
            "{}",
            p.peak_decode_tps
        );
    }

    #[test]
    fn closing_view_rescues_request_row_on_chained_completion() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, false, 0, 0, 0), t0);
        p.observe(&vllm_slot(5, true, 0, 0, 0, 0.0, 0.0, None), t0 + step);
        // A (task 5: 1000 prompt / 100 cached / 40 generated / 2.0 s
        // TTFT / 0.05 s ITL total) closes and B (task 6) is admitted in
        // the same poll. The baseline re-anchors onto B, so the closing
        // view is the only place A's counters survive.
        p.observe(
            &vllm_slot(
                6,
                true,
                0,
                0,
                0,
                0.0,
                0.0,
                Some(crate::observe::ClosingRequest {
                    id_task: 5,
                    prompt: 1000,
                    cached: 100,
                    gen: 40,
                    prefill_secs: None,
                    measured_prefill_tps: Some(450.0),
                    measured_decode_tps: None,
                    ttft_secs: 2.0,
                    itl_sum: 0.05,
                    published_prompt: 0,
                    published_gen: 0,
                    published_completions: 1,
                    span_known: 0,
                    exact_prompt: 0,
                    exact_gen: 0,
                }),
            ),
            t0 + step + Duration::from_secs(17),
        );
        // A's row exists and is complete despite the re-anchor.
        let a = p
            .history
            .iter()
            .rev()
            .find(|r| r.id_task == 5)
            .expect("the finished request keeps a row");
        assert_eq!(a.prefill_tokens, 900, "prompt minus cache");
        assert_eq!(a.decoded, 40);
        assert!(
            (a.avg_prefill_tps() - 450.0).abs() < 1.0,
            "{}",
            a.avg_prefill_tps()
        );
        assert!((a.avg_decode_tps() - 39.0 / 0.05).abs() < 1.0);
        // B is measured against the re-anchored baseline.
        p.observe(
            &vllm_slot(6, false, 50, 50, 20, 0.5, 0.1, None),
            t0 + step * 2 + Duration::from_secs(17),
        );
        let b = p
            .history
            .iter()
            .rev()
            .find(|r| r.id_task == 6)
            .expect("B's row");
        assert_eq!(b.prefill_tokens, 50);
        assert_eq!(b.decoded, 20);
        assert!((b.avg_prefill_tps() - 100.0).abs() < 1.0, "50 / 0.5 s",);
        assert_eq!(p.session_prefilled, 950, "900 (A) + 50 (B)");
        assert_eq!(p.session_decoded, 60);
        assert_eq!(p.history.len(), 2);
    }

    #[test]
    fn rescued_request_without_ttft_is_still_closed() {
        // A closing view with no usable TTFT (aborted request, or a
        // build without the histogram) must still stamp `ended`, or the
        // row stays "live" forever.
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, false, 0, 0, 0), t0);
        p.observe(&vllm_slot(5, true, 0, 0, 0, 0.0, 0.0, None), t0 + step);
        p.observe(
            &vllm_slot(
                6,
                true,
                0,
                0,
                0,
                0.0,
                0.0,
                Some(crate::observe::ClosingRequest {
                    // The close names the *finished* request — task 5 — even
                    // though a successor (task 6) lit the same scrape. The
                    // row it rescues must still close (no TTFT: a row with
                    // no usable span must not stay "live" forever).
                    id_task: 5,
                    prompt: 100,
                    cached: 0,
                    gen: 10,
                    prefill_secs: None,
                    measured_prefill_tps: None,
                    measured_decode_tps: None,
                    ttft_secs: 0.0,
                    itl_sum: 0.0,
                    published_prompt: 0,
                    published_gen: 0,
                    published_completions: 1,
                    span_known: 0,
                    exact_prompt: 0,
                    exact_gen: 0,
                }),
            ),
            t0 + step * 2,
        );
        let a = p.history.iter().find(|r| r.id_task == 5).expect("row kept");
        assert!(a.ended.is_some(), "rescued row must not stay live");
    }

    #[test]
    fn rate_window_smooths_bursts_and_decays() {
        let t0 = Instant::now();
        let mut w = RateWindow::new(Duration::from_millis(1000));
        let ms = |m: u64| t0 + Duration::from_millis(m);
        for (i, n) in [8, 0, 8, 0, 8].iter().enumerate() {
            let a = i as u64 * 200;
            w.push(ms(a), ms(a + 200), *n);
        }
        // 24 tokens over the full second, not 40 during the burst samples.
        assert!(
            (w.rate(ms(1000)) - 24.0).abs() < 0.1,
            "{}",
            w.rate(ms(1000))
        );
        for i in 5..12u64 {
            w.push(ms(i * 200), ms(i * 200 + 200), 0);
        }
        assert_eq!(w.rate(ms(2400)), 0.0);
    }

    #[test]
    fn spec_stats_windowed_acceptance() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let m = |d: u64, a: u64, s: u64| SpecMetrics {
            draft_tokens: d,
            accepted: a,
            verify_steps: s,
            n_decode: 0,
            tokens_predicted: a + s,
            busy_secs: 0.0,
        };
        p.observe_spec(&m(100, 60, 100), t0);
        p.observe_spec(&m(120, 75, 120), t0 + Duration::from_millis(200));
        p.observe_spec(&m(140, 85, 140), t0 + Duration::from_millis(400));
        assert!(p.spec.available);
        // 25 accepted of 40 drafted in the window.
        assert!(
            (p.spec.accept_rate - 0.625).abs() < 1e-3,
            "{}",
            p.spec.accept_rate
        );
        assert!((p.spec.mean_accepted - 0.625).abs() < 1e-3);
        assert!((p.spec.session_accept_rate() - 85.0 / 140.0).abs() < 1e-4);
        assert_eq!(p.spec.accept_hist.len(), 2);
    }

    #[test]
    fn spec_stats_completion_only_counters() {
        // Strata: the whole request's drafts land on one poll, with the
        // server's decode time to spread them over.
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let m = |d: u64, a: u64, s: u64, busy: f64| SpecMetrics {
            draft_tokens: d,
            accepted: a,
            verify_steps: s,
            n_decode: 0,
            tokens_predicted: a + s,
            busy_secs: busy,
        };
        p.observe_spec(&m(0, 0, 0, 0.0), t0);
        p.observe_spec(&m(300, 150, 100, 5.0), t0 + Duration::from_millis(400));
        assert!((p.spec.accept_rate - 0.5).abs() < 1e-3);
        assert!(
            (p.spec.steps_per_sec - 20.0).abs() < 0.5,
            "{}",
            p.spec.steps_per_sec
        );
        // Idle polls past the window keep the last acceptance.
        p.observe_spec(&m(300, 150, 100, 5.0), t0 + Duration::from_secs(3));
        assert_eq!(p.spec.drafts_per_sec, 0.0);
        assert!((p.spec.accept_rate - 0.5).abs() < 1e-3);
    }

    #[test]
    fn meter_holds_peak_and_autoscales() {
        let t0 = Instant::now();
        let mut m = Meter::auto(10.0);
        assert_eq!(m.scale(), 10.0);
        m.update(40.0, t0, 0.2);
        assert_eq!(m.scale(), 40.0);
        assert_eq!(m.frac(), 1.0);
        m.update(4.0, t0 + Duration::from_millis(500), 0.2);
        assert_eq!(m.hold, 40.0, "held");
        m.update(4.0, t0 + Duration::from_millis(2000), 0.5);
        assert!(m.hold < 40.0 && m.hold >= 4.0, "falls to {}", m.hold);
        let f = Meter::fixed(100.0);
        assert_eq!(f.scale(), 100.0);
        assert_eq!(pcie_link_mb_s(3, 16), Some(985.0 * 16.0));
        assert_eq!(pcie_link_mb_s(0, 16), None);
    }

    #[test]
    fn host_rates_from_counter_deltas() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let s = |disk: u64, proc_rd: u64, flt: u64, rx: f32| HostSample {
            disk_read_bytes: Some(disk),
            proc_read_bytes: Some(proc_rd),
            proc_majflt: Some(flt),
            pcie_mb_s: vec![(0, rx, 1.0), (1, rx / 2.0, 0.0)],
            pcie_ok: true,
            ..Default::default()
        };
        p.observe_host(&s(1_000_000_000, 500_000_000, 100, 10.0), t0);
        p.observe_host(
            &s(1_200_000_000, 550_000_000, 150, 800.0),
            t0 + Duration::from_millis(200),
        );
        assert!(
            (p.bw.disk.value - 1000.0).abs() < 1.0,
            "{}",
            p.bw.disk.value
        );
        assert!((p.bw.proc_disk_mb_s - 250.0).abs() < 1.0);
        assert!((p.bw.majflt_per_s - 250.0).abs() < 1.0);
        assert_eq!(p.bw.pcie_rx.len(), 2);
        assert_eq!(p.bw.pcie_rx[0].value, 800.0);
        assert_eq!(p.bw.pcie_rx[1].value, 400.0);
        // A garbage dmon sample is dropped: the meter keeps its last reading.
        p.observe_host(
            &s(1_200_000_000, 550_000_000, 150, 209_688.0),
            t0 + Duration::from_millis(400),
        );
        assert_eq!(p.bw.pcie_rx[0].value, 800.0);
        assert_eq!(p.bw.pcie_rx[0].hold, 800.0);
        p.bw.pcie_rx[0].full_scale = Some(7_880.0);
        p.observe_host(
            &s(1_200_000_000, 550_000_000, 150, 9_000.0),
            t0 + Duration::from_millis(600),
        );
        assert_eq!(p.bw.pcie_rx[0].value, 800.0, "above the link cap: dropped");
        p.observe_host(
            &s(1_200_000_000, 550_000_000, 150, 7_000.0),
            t0 + Duration::from_millis(800),
        );
        assert_eq!(p.bw.pcie_rx[0].value, 7_000.0);
    }

    #[test]
    fn weight_streams_follow_step_rate() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, true, 100, 100, 0), t0);
        p.observe(&slot(1, true, 100, 100, 10), t0 + step);
        p.observe(&slot(1, true, 100, 100, 20), t0 + step * 2);
        assert_eq!(p.phase, Phase::Decode);
        let layout = WeightLayout {
            known: true,
            total_bytes: 20_000_000_000,
            active_bytes: 2_500_000_000,
            cpu_bytes: 8_000_000_000,
            ubatch: 512,
        };
        // 8 GB on the CPU side × 12.5 % active = 1 GB per step from RAM,
        // the other 1.5 GB per step from VRAM.
        let (ram_b, vram_b) = layout.per_step();
        assert!((ram_b - 1.0e9).abs() < 1.0);
        assert!((vram_b - 1.5e9).abs() < 1.0);
        p.tick_bandwidth(&layout, t0 + step * 2, 0.2);
        let steps = p.bw.steps_per_s;
        assert!(steps > 0.0);
        assert!(
            (p.bw.ram.value - steps).abs() < 1e-3,
            "ram {} steps {}",
            p.bw.ram.value,
            steps
        );
        assert!((p.bw.vram.value - 1.5 * steps).abs() < 1e-3);
    }

    #[test]
    fn unified_pool_fill_drives_the_pool_meter() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let g = || GpuStats {
            index: 0,
            unified: true,
            mem_total_mb: 120_000,
            mem_used_mb: 60_000,
            utilization_mem: 0.0,
            ..Default::default()
        };
        p.observe_gpu(&[g()], t0);
        assert!(
            p.bw.vram_busy[0].value < 1.0,
            "no controller on a unified part"
        );
        assert!(
            (p.bw.vram_pool[0].value - 50.0).abs() < 0.1,
            "{}",
            p.bw.vram_pool[0].value
        );
        // A discrete card leaves the pool meter at zero.
        let d = GpuStats {
            index: 1,
            unified: false,
            mem_total_mb: 24_000,
            mem_used_mb: 20_000,
            utilization_mem: 80.0,
            ..Default::default()
        };
        p.observe_gpu(&[g(), d], t0 + Duration::from_millis(200));
        assert!(p.bw.vram_pool[1].value < 1.0);
        assert!((p.bw.vram_busy[1].value - 80.0).abs() < 0.1);
    }

    #[test]
    fn peak_hold_decays_after_delay() {
        let now = Instant::now();
        let mut pk = (0.0f32, now);
        hold_peak(&mut pk, 0.9, now, 0.2);
        assert_eq!(pk.0, 0.9);
        hold_peak(&mut pk, 0.2, now + Duration::from_millis(500), 0.2);
        assert_eq!(pk.0, 0.9, "held");
        hold_peak(&mut pk, 0.2, now + Duration::from_millis(2000), 0.5);
        assert!(pk.0 < 0.9 && pk.0 >= 0.2, "decayed to {}", pk.0);
    }
}
