//! mlx-serve metrics adapter (Apple MLX inference servers).
//!
//! mlx-serve (and `mlx_lm.server`-shaped servers) expose vLLM's Prometheus
//! names on `/metrics` — `prompt_tokens_total`, `generation_tokens_total`,
//! `request_success_total`, the TTFT/latency histograms — so detection names
//! the engine from the `owned_by: "mlx-serve"` entry on its `v1/models` and
//! the poller drives the shared [`crate::vllm::VllmAdapter`] from
//! [`crate::vllm::poll_vllm`]. Those counters move at request completion:
//! between completions the panels hold their last values. The engine's own
//! `mlx_serve:` namespace (live prefill/decode gauges, GPU utilisation,
//! per-request histograms) is richer but races the pipeline in ways the
//! simple counters do not; a v2 may ride those too.
//!
//! This file's own job is identity: mlx-serve's `/v1/models` has no `root`,
//! so the weights directory — what the model-info and layers panels read —
//! comes from the `--model` flag on the command line, and the name the
//! metrics carry must match the served id for the cross-wiring guard.

use crate::observe::{http_get, HttpAuth, LiveStats};
use crate::vllm::{cross_wired, parse_vllm_metrics, VllmCounters};

/// mlx-serve's default HTTP port.
pub const DEFAULT_PORT: u16 = 8080;

/// What the engine's `/v1/models` entry says about itself: served id,
/// context limit, and the MTP panel's static state. mlx-serve publishes
/// `meta.mtp_loaded`/`mtp_available` — the drafter is in the weights and
/// can run — but no speculative *counters*, so the adapter reports the
/// type and depth from this and leaves the counters absent. The panel then
/// shows "drafting on" with the honest missing-metrics note (the same
/// `available == false` shape a llama.cpp server without `/metrics` uses)
/// instead of the false "the server is not drafting" that a blank
/// `spec_types` produces while MTP is very much loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServedIdentity {
    pub id: String,
    pub ctx: usize,
    /// `meta.mtp_loaded` (or `mtp_available`): the model carries MTP layers
    /// and the engine says they are available.
    pub mtp: bool,
}

/// The engine's own `/v1/models` answer. `None` unless the body is a
/// `v1/models` list served by an MLX engine.
pub fn parse_served_model(body: &str) -> Option<ServedIdentity> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let first = v.get("data")?.as_array()?.first()?;
    let owned_by = first.get("owned_by").and_then(|x| x.as_str())?;
    if owned_by != "mlx-serve" && owned_by != "mlx_lm.server" {
        return None;
    }
    let id = first.get("id").and_then(|x| x.as_str())?;
    if id.is_empty() {
        return None;
    }
    let ctx = first
        .get("max_model_len")
        .and_then(|x| x.as_u64())
        .or_else(|| first.get("context_length").and_then(|x| x.as_u64()))
        .unwrap_or(0) as usize;
    let meta = first.get("meta");
    let flag = |name: &str| {
        meta.and_then(|m| m.get(name))
            .and_then(|x| x.as_bool())
            .unwrap_or(false)
    };
    Some(ServedIdentity {
        id: id.to_string(),
        ctx,
        mtp: flag("mtp_loaded") || flag("mtp_available"),
    })
}

/// Ask `/v1/models` (un-keyed on mlx-serve) for the served id and context.
/// Detection-time and explicit-endpoint only; it never runs per poll.
pub async fn poll_served_model(host: &str, port: u16, auth: &HttpAuth) -> Option<ServedIdentity> {
    let body = http_get(host, port, "/v1/models", auth).await.ok()?;
    parse_served_model(&body)
}

/// Sync entry for the process-scan pass in `model_detect`, which runs both
/// inside and outside a tokio runtime and cannot `block_on` either way; the
/// probe therefore runs on its own thread. When the process was only just
/// scanned, `wait` gives a starting server a moment to begin answering.
pub fn served_model_blocking(
    host: &str,
    port: Option<u16>,
    wait: std::time::Duration,
) -> Option<ServedIdentity> {
    let host = host.to_string();
    let port = port.unwrap_or(DEFAULT_PORT);
    std::thread::spawn(move || {
        let auth = HttpAuth::default();
        if wait > std::time::Duration::ZERO {
            std::thread::sleep(wait);
        }
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?
            .block_on(poll_served_model(&host, port, &auth))
    })
    .join()
    .ok()
    .flatten()
}

/// mlx-serve's live in-flight gauges, the one part of its own namespace
/// that is race-free enough to ride without request-identity tracking:
/// the token counters move every decode step, the expected-prompt counter
/// the moment prefill starts.
#[derive(Debug, Default, Clone)]
pub struct LiveGauges {
    /// `mlx_serve:generation_tokens_live` — decode tokens issued so far,
    /// moving per step while the request is in flight.
    pub generation_live: f64,
    /// `mlx_serve:prefill_tokens_live` — prompt tokens already forwarded
    /// through prefill; 0 between prefills.
    pub prefill_live: f64,
    /// `mlx_serve:prefill_tokens_expected` — the in-flight request's full
    /// prompt size (excludes prefix-cache restores), published at prefill
    /// start; 0 when nothing is prefilling.
    pub prefill_expected: f64,
    /// `mlx_serve:prefill_tokens_total` — the engine's lifetime prefill
    /// counter, moving per step *while prefill runs*: the only measured
    /// in-flight prefill rate the server offers.
    pub prefill_total: f64,
    /// `mlx_serve:requests_prefilling` — the engine's own yes/no on a
    /// prefill being in progress right now.
    pub requests_prefilling: bool,
    /// `mlx_serve:batched_group_size` — the open-batch countdown: requests
    /// still expected before a batched prefill starts. Observed staying 0
    /// through whole prefills on the live engine, so nothing keys on it.
    pub batch_wait: f64,
}

/// Read the three live gauges out of a metrics body. Absent lines stay 0 —
/// an engine that stops publishing them degrades to pure completion
/// counters rather than going negative.
pub fn parse_live(body: &str) -> LiveGauges {
    let mut g = LiveGauges::default();
    for line in body.lines() {
        let Some(rest) = line.strip_prefix("mlx_serve:") else {
            continue;
        };
        let Some((head, value)) = rest.trim_end().rsplit_once(' ') else {
            continue;
        };
        let Ok(value) = value.parse::<f64>() else {
            continue;
        };
        match head.split('{').next().unwrap_or(head) {
            "generation_tokens_live" => g.generation_live = value,
            "prefill_tokens_live" => g.prefill_live = value,
            "prefill_tokens_expected" => g.prefill_expected = value,
            "prefill_tokens_total" => g.prefill_total = value,
            "requests_prefilling" => g.requests_prefilling = value > 0.5,
            "batched_group_size" => g.batch_wait = value,
            _ => {}
        }
    }
    g
}

/// GET `/metrics` and keep only our engine's counters: vLLM's cross-wiring
/// guard plus the identity check that `parse_vllm_metrics` alone cannot
/// make — mlx-serve labels its metrics with the config's architecture name
/// (`model_name="qwen4_exp"`), not the served id, so the shared guard's
/// prefix test would not match and a foreign mlx-serve sharing the port
/// would be accepted. Requiring the served id on `v1/models` keeps the
/// scrape bound to this process's own engine.
pub async fn poll(
    host: &str,
    port: u16,
    expected_name: &str,
    others: &[(String, u16)],
    auth: &HttpAuth,
) -> Option<(VllmCounters, LiveGauges)> {
    let served = poll_served_model(host, port, auth).await?;
    if cross_wired(&served.id, expected_name, port, others) {
        return None;
    }
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
    let counters = parse_vllm_metrics(&body)?;
    Some((counters, parse_live(&body)))
}

/// Per-slot live reconstruction from the engine's in-flight gauges, folded
/// into the shared adapter's view. The vLLM-named counters move once per
/// request, so between completions nothing moves and the panels freeze; these
/// gauges move per step, and the engine itself says what it is doing:
/// `requests_prefilling` while it prefilles, `prefill_tokens_live`/
/// `prefill_tokens_expected` for the in-flight prefill,
/// `generation_tokens_live` for decode.
///
/// What the live side can prove, it fills in: `decoded` rides the decode
/// gauge, and the prefill's progress is `prefill_tokens_total`'s movement
/// across the window the engine kept lit (`vllm:prompt_tokens_total` only
/// moves at completion, which is why perf.rs's windowed rate divides a whole
/// prompt by one poll interval and reads zero). The completion scrape passes
/// through untouched — the exact counters always win. What it cannot prove
/// stays quiet: the prompt's cached share is invisible to the live gauges
/// (shown from the last completed request until the completion makes it
/// exact), and mid-prefill rate readings are quantized to the poll, while
/// perf.rs's measured-TTFT path reports the request's exact rate at
/// completion. A gauge rewinding means the engine restarted.
pub struct LiveRates {
    running: bool,
    gen_base: f64,
    /// The prefill window: `prefill_total_base` is `prefill_tokens_total`
    /// as of the previous scrape, `prefill_total_lit` the computed prompt
    /// the engine has published for the request in flight (a step function:
    /// the expectation arrives whole at the prefill's first publish),
    /// `prefill_expected` the same expectation as last seen — a new value
    /// above it is a new request, which opens a fresh window;
    /// `prefill_total_secs` the span of scrapes since the window opened
    /// (0.0 = no window). Edge quantization is one poll, ~100 ms.
    prefill_total_base: f64,
    prefill_total_lit: f64,
    prefill_expected: f64,
    prefill_total_secs: f32,
    /// The rate's base: `prefill_tokens_total` as of the scrape the last
    /// in-flight rate published (or the window opened), `rate_secs` the
    /// span since that scrape. The rate is the step counter's movement
    /// since the base over the span since the base — the engine's own
    /// publication, held across the publish cadence's dark beats.
    rate_base: f64,
    rate_secs: f32,
    /// The last published in-flight rate (tokens/s): the step counter's
    /// movement since the last publish over the span since the last pub-
    /// lish, re-published every scrape while a window is open so the panel
    /// holds its reading through the counter's dark beats instead of blink-
    /// ing. Zero when no reading exists; retired with the request.
    prefill_rate: f64,
    /// `prompt_tokens` and the cached share from the last completed request:
    /// the prefix-restored tokens are invisible to the live gauges, so an
    /// in-flight request shows computed-so-far + last request's cache.
    last_prompt: usize,
    last_cached: usize,
    /// `prefill_tokens_live` as of the previous scrape (the engine's block
    /// publication; nothing divides by it, it exists to open windows).
    prefill_live_base: f64,
    /// Detection's answer from the engine's `/v1/models`: the model carries
    /// MTP layers and the engine says they are available. mlx-serve pub-
    /// lishes no speculative counters, so the shared adapter's identity
    /// guard blanks `spec_types` to "none" every scrape; this tracker —
    /// built only for mlx slots — keeps the engine's own truth intact.
    mtp: bool,
}

impl Default for LiveRates {
    fn default() -> Self {
        Self {
            running: false,
            gen_base: 0.0,
            prefill_total_base: 0.0,
            prefill_total_lit: 0.0,
            prefill_expected: 0.0,
            prefill_total_secs: 0.0,
            rate_base: 0.0,
            rate_secs: 0.0,
            prefill_rate: 0.0,
            last_prompt: 0,
            last_cached: 0,
            prefill_live_base: 0.0,
            mtp: false,
        }
    }
}

/// The rate tracker for one mlx slot. `mtp` is `ServedIdentity::mtp` from
/// detection (false when the server never answered `/v1/models`).
pub fn live_rates(mtp: bool) -> LiveRates {
    LiveRates {
        mtp,
        ..LiveRates::default()
    }
}

impl LiveRates {
    /// Fold one scrape into the slot, enriching the shared adapter's view
    /// with the live token counts and in-flight prefill rate. The closing
    /// sample passes through untouched: the completion counters are exact.
    pub fn observe(&mut self, c: &VllmCounters, g: &LiveGauges, stats: &mut LiveStats) {
        if g.generation_live < self.gen_base - 0.5
            || g.prefill_total < self.prefill_total_base - 0.5
        {
            // The engine restarted: all window state resets; the identity
            // detection read from /v1/models still stands (this is the same
            // port we are reconnecting to).
            *self = live_rates(self.mtp);
        }
        if stats.processing && !self.running {
            // Fresh admission (or first sight of a request): base at the
            // previous sample, mirroring the vLLM adapter. The vllm-named
            // counters sit frozen for the request's whole lifetime, so the
            // live gauges carry everything the panels move on in between.
            self.gen_base = c.generation_total.max(g.generation_live);
        }

        let lit = g.prefill_expected > 0.0 || g.prefill_live > 0.0;
        let completed = stats.processing && !self.running;

        // mlx-serve says `mtp_loaded` on /v1/models but publishes no specu-
        // lative counters at all, so the shared adapter answers every scrape
        // with "none" (no drafts to key identity on) and the panel reads
        // "the server is not drafting" while MTP is very much loaded. Every
        // scrape this tracker sees belongs to the mlx slot the server said
        // it drafted for, so the identity's word wins here: the panel's
        // `available == false` path then shows "drafting on" with the
        // honest missing-metrics note.
        if self.mtp && (stats.spec_types.is_empty() || stats.spec_types == "none") {
            stats.spec_types = "mtp".to_string();
        }

        if stats.processing {
            let live_decoded = (g.generation_live - self.gen_base).max(0.0) as usize;
            stats.decoded = stats.decoded.max(live_decoded);

            // The window model, as the live server publishes it (traced on
            // the M5 Max engine through its whole lifecycle). `requests_
            // prefilling` is lit through the request's prefill phase and
            // dark once it decodes. `prefill_tokens_total` — the engine's
            // per-step counter — moves every scrape while the engine
            // prefills and holds its value once decode begins. The pair
            // (`prefill_tokens_live`/`prefill_tokens_expected`) publishes
            // the progress line: the expectation — the prompt's exact size
            // — lands whole, the live gauge steps toward it and goes dark
            // when the prefill closes. A window opens when a new
            // expectation lands (its value tells requests apart: the engine
            // keeps one publish block per metric, so a successor's
            // expectation can share a scrape with an old request's
            // completion) or when the phase bit lights on a fresh
            // admission, and runs until the request's completion scrape
            // closes it.
            let prefilling = g.requests_prefilling;
            // Window bookkeeping: `prefill_total_secs` is this request's
            // open span (0.0 = no window); `rate_base` is the step counter
            // as of the scrape the last rate published (or the window
            // opened), `rate_secs` the span since that scrape. The rate is
            // movement since the base over the span since the base: the
            // engine's own publication, held across the publish cadence's
            // dark beats (a rate that blinks every scrape is no rate). The
            // expectation's whole-size landing is a value, never a division
            // input — the base starts at the landing scrape's value, so the
            // first reading divides the prefill's first steps across the
            // first whole scrape-to-scrape interval. A window opens on the
            // expectation's landing or the phase bit on a fresh admission
            // and runs to the request's completion scrape.
            let new_request =
                g.prefill_expected > 0.0 && g.prefill_expected != self.prefill_expected;
            // "Freshly admitted" is the adapter's answer — processing this
            // scrape, not last — since a live server has been processing
            // long before this request's expectation ever lands.
            if new_request {
                self.prefill_total_lit = 0.0;
                self.prefill_live_base = g.prefill_live;
                self.prefill_total_secs = 0.1;
                // The expectation's whole-size landing is never a division
                // input: the base starts at the landing scrape's value, so
                // the first reading measures the engine's own steps from the
                // landing on, across the span from the landing on. The
                // landing scrape itself publishes no reading.
                self.rate_base = g.prefill_total;
                self.rate_secs = 0.0;
                self.prefill_rate = 0.0;
            } else if self.prefill_total_secs == 0.0 && prefilling {
                // The phase bit on a fresh admission with nothing published
                // yet: open the window early so the prefill's first steps
                // are measuring from its first scrape.
                self.prefill_total_lit = 0.0;
                self.prefill_live_base = g.prefill_live;
                self.prefill_total_secs = 0.1;
                // Same half-beat rule as the expectation's landing: the
                // phase bit flips between polls, so the opening scrape
                // charges no span and its own step is a value, never a
                // division input — the base sits at it, and the first
                // reading divides one whole scrape-to-scrape interval of
                // movement made after the window opened. The opening scrape
                // itself publishes no reading.
                self.rate_base = g.prefill_total;
                self.rate_secs = 0.0;
                self.prefill_rate = 0.0;
            } else if self.prefill_total_secs > 0.0 {
                self.prefill_total_secs += 0.1;
            }
            // The publication: movement since the base over the span since
            // the base. A scrape that moves the counter republishes; one
            // that moves nothing holds the last reading. The publish con-
            // sumes the interval it divided (the base moves to this scrape),
            // so the next reading measures the engine's next interval — a
            // true step-by-step rate, not a diluting whole-span mean. The
            // span charges on every scrape of the open window; the expec-
            // tation's landing scrape itself is half a poll (the engine's
            // publish lands between two scrapes), so the first interval
            // needs the next scrape to fill a whole 0.1 s before anything
            // divides — the reading then reports the engine's steps at
            // their true per-second rate.
            if self.prefill_total_secs > 0.0 {
                // The span charges on every scrape past the window's open-
                // ing (which is a half-beat: its publication lands between
                // polls). 0.0 = no reading can divide yet.
                if self.prefill_total_secs > 0.1 {
                    self.rate_secs += 0.1;
                }
                // Movement since the last publish (or since the window
                // opened — the base anchors at that scrape's value, so the
                // expectation's whole-size landing and the phase bit's an-
                // nounced step are both values, never division inputs).
                let since_rate = (g.prefill_total - self.rate_base).max(0.0);
                if since_rate > 0.0 && self.rate_secs >= 0.1 {
                    // The reading: that movement over the span since the
                    // last publish. The opening half-beat is already out of
                    // the span (the window charges only past its opening
                    // scrape), so the first reading divides one whole
                    // scrape-to-scrape interval and every reading after it
                    // too. A dark beat (the step counter holds) charges the
                    // span without new movement: the next reading dilutes
                    // honestly across it — a rate that blinks mid-prefill
                    // is no rate. The publish consumes the span it divided.
                    self.prefill_rate = since_rate / self.rate_secs as f64;
                    self.rate_base = g.prefill_total;
                    self.rate_secs = 0.0;
                }
                if self.prefill_rate > 0.0 {
                    stats.prefill_tps = Some(self.prefill_rate as f32);
                    // An in-flight reading from the engine's own step counter.
                    stats.prefill_rate_live = true;
                }
            }

            if lit {
                self.prefill_total_lit = self
                    .prefill_total_lit
                    .max(g.prefill_live)
                    .max(g.prefill_expected);
            }

            // The prompt the panels show: the engine's latest publication —
            // the exact prompt size the moment the expectation lands, the
            // block gauge's line before and after it. A request with
            // nothing published yet claims one processed token: perf.rs
            // infers the phase from the counters alone, and mlx's publish
            // cadence keeps them frozen for seconds at a stretch (the same
            // at-least-one convention strata.rs uses).
            if self.prefill_total_lit > 0.0 {
                stats.prompt_processed = stats
                    .prompt_processed
                    .max(self.prefill_total_lit as usize)
                    .max(1);
            } else if stats.decoded > 0 || self.prefill_total_secs > 0.0 {
                // A window open with nothing published yet, or a decoding
                // request whose prompt the completion counters have not
                // named: claim one processed token so the inferred phase is
                // Prefill, not Decode (strata's convention — the panel must
                // not blank its prefill numeral while the engine works).
                stats.prompt_processed = stats.prompt_processed.max(1);
            }

            if stats.prompt_processed > 0 {
                // The prompt the panels show: the live request's cached
                // share is invisible to the live gauges until the request
                // closes, so it rides in from the last completed request.
                stats.prompt_tokens = stats
                    .prompt_tokens
                    .max(self.last_cached + stats.prompt_processed)
                    .max(self.last_prompt);
            }

            // The panel's phase, settled by the engine's own phase bit.
            // perf.rs infers it from counter movement, but mlx publishes
            // its metrics in blocks — the step counter can hold one value
            // across a whole dark stretch of the publish cadence — so the
            // windowed movement stalls mid-prefill and the inference flips
            // to Decode, blanking the big prefill numeral. The engine
            // answers "am I prefilling" directly; on mlx that beats the
            // inference: while it says prefill, `decoded` cannot be
            // allowed to move the phase, so it reports none. Completion's
            // exact counters (and the decode gauges once the bit goes dark)
            // take the panel back.
            if g.requests_prefilling {
                stats.decoded_present = false;
            }
        }

        // The request is over: its window (and the published progress line
        // it carried) ends with it. The completion scrape still charges
        // the span once more (the engine's movement on this very scrape,
        // clamped to the request's own span delta, over the span including
        // this beat), so the last in-flight reading is honest even when
        // the publish cadence went dark for the prefill's tail. From the
        // completion on, perf.rs's measured-TTFT path carries the
        // request's exact prefill mean.
        if completed || !stats.processing {
            // The in-flight reading ends with the request: perf.rs carries
            // the exact whole-span mean from the completion on.
            self.prefill_total_secs = 0.0;
            self.prefill_total_lit = 0.0;
            self.prefill_expected = 0.0;
            self.rate_base = 0.0;
            self.rate_secs = 0.0;
            self.prefill_rate = 0.0;
        }

        self.running = stats.processing;
        self.prefill_total_base = g.prefill_total;
        self.prefill_expected = g.prefill_expected;
        if stats.prompt_tokens > 0 {
            self.last_prompt = stats.prompt_tokens;
        }
        if stats.prompt_processed > 0 && !stats.processing {
            // The request that just closed is the last one whose cached
            // share the completion counters made exact.
            self.last_cached = stats.cache_tokens;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_identity_from_a_real_mlx_models_body() {
        // Shape captured from mlx-serve's /v1/models (no `root`, ctx on
        // both keys, capabilities irrelevant here).
        let body = r#"{"object":"list","data":[{"id":"Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit",
            "object":"model","owned_by":"mlx-serve","loaded":true,"state":"ready",
            "context_length":204800,"max_model_len":204800,"capabilities":["chat"]}]}"#;
        let served = parse_served_model(body).unwrap();
        assert_eq!(served.id, "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit");
        assert_eq!(served.ctx, 204800);
        assert!(!served.mtp, "no meta block means no claim of MTP");

        // A vLLM answer is not ours: detection must keep it on the vllm path.
        let vllm =
            r#"{"object":"list","data":[{"id":"LFM","owned_by":"vllm","root":"/models/LFM"}]}"#;
        assert!(parse_served_model(vllm).is_none());

        // mlx_lm.server is the same engine family.
        let mlx_lm = r#"{"data":[{"id":"m","owned_by":"mlx_lm.server"}]}"#;
        let served = parse_served_model(mlx_lm).unwrap();
        assert_eq!((served.id.as_str(), served.ctx), ("m", 0));

        // The live server's meta block: MTP loaded, no drafter model. The
        // panel shows "drafting on" from this even though the metrics
        // carry no speculative counters.
        let with_mtp = r#"{"data":[{"id":"m","owned_by":"mlx-serve","max_model_len":204800,
            "meta":{"architecture":"qwen4_exp","mtp_loaded":true,"mtp_available":true,
                    "drafter_loaded":false,"spec_exact":false}}]}"#;
        let served = parse_served_model(with_mtp).unwrap();
        assert!(served.mtp);

        assert!(parse_served_model("not json").is_none());
        assert!(parse_served_model(r#"{"data":[]}"#).is_none());
    }

    /// One scrape's worth of engine state, in the shape the live server
    /// publishes it.
    #[derive(Clone, Copy)]
    struct Scrape {
        gen_total: f64,
        succeeded: f64,
        /// `vllm:num_requests_running`: 1 while a request is in flight, 0
        /// at idle (mlx-serve publishes it honestly; a plain HTTP/1.0 probe
        /// once read a stale 1 and sent this whole model down a wrong path).
        running: bool,
        gen_live: f64,
        pre_live: f64,
        expected: f64,
        pre_total: f64,
        prefilling: bool,
    }

    fn live(s: Scrape) -> LiveGauges {
        let body = format!(
            "mlx_serve:generation_tokens_live {}\n\
             mlx_serve:prefill_tokens_live {}\n\
             mlx_serve:prefill_tokens_expected {}\n\
             mlx_serve:prefill_tokens_total {}\n\
             mlx_serve:requests_prefilling {}\n\
             mlx_serve:batched_group_size 0\n\
             mlx_serve:gpu_utilization_pct 66\n",
            s.gen_live, s.pre_live, s.expected, s.pre_total, s.prefilling as u8
        );
        parse_live(&body)
    }

    fn counters(s: Scrape) -> VllmCounters {
        VllmCounters {
            generation_total: s.gen_total,
            succeeded: s.succeeded,
            running: if s.running { 1.0 } else { 0.0 },
            // mlx-serve moves `vllm:prompt_tokens_total` at completion,
            // and `vllm:num_requests_running` is honestly published as 0
            // at idle / 1 in flight (the stuck-1 readings an earlier probe
            // caught were stale connections). Prompts ride `pre_total`.
            prompt_total: s.pre_total,
            ..Default::default()
        }
    }

    #[test]
    fn parses_the_live_gauges() {
        let g = parse_live(
            "mlx_serve:generation_tokens_live 81593\n\
             mlx_serve:prefill_tokens_expected 1075\n\
             mlx_serve:requests_prefilling 1\n",
        );
        assert_eq!(g.generation_live, 81593.0);
        assert_eq!(g.prefill_expected, 1075.0);
        assert!(g.requests_prefilling);
        // An unrelated body degrades to zeros, not negatives.
        let g = parse_live("vllm:generation_tokens_total 5\n");
        assert_eq!(
            (g.generation_live, g.prefill_live, g.prefill_expected),
            (0.0, 0.0, 0.0)
        );
    }

    /// A request's whole lifetime at a 100 ms poll: the vllm-named counters
    /// sit frozen from admission to completion while the live gauges move per
    /// step — and the panels must show movement on every single poll.
    /// A request's whole lifetime at a 100 ms poll, replaying the shapes the
    /// live server publishes: the vllm-named counters sit frozen from
    /// admission to completion (with them alone the panels freeze), the
    /// engine's live prefill pair publishes a beat behind its own steps
    /// (lit/dark flutters), and the expectation — the prompt's exact size —
    /// arrives whole at the prefill's first publish. No poll may show
    /// nothing, and no
    /// poll may double-count when the completion scrape also carries the
    /// successor's lit prefill: the race that made the parked adapter a
    /// request-identity state machine.
    #[test]
    fn live_gauges_keep_every_poll_moving() {
        let mut lr = LiveRates::default();
        let mut adapter = crate::vllm::VllmAdapter::new();
        let base = Scrape {
            gen_total: 81452.0,
            succeeded: 300.0,
            running: false,
            gen_live: 81452.0,
            pre_live: 0.0,
            expected: 0.0,
            pre_total: 400000.0,
            prefilling: false,
        };
        let (mut s, _) = adapter.observe(&counters(base));
        lr.observe(&counters(base), &live(base), &mut s);
        assert!(!s.processing);

        // Request admitted; it prefills a 1075-token computed prompt, then
        // decodes. `pre_total` is the engine's lifetime prefill counter;
        // `pre_live` is the block gauge, published whole-blocks (live: 8192
        // blocks; a prompt under one block shows the whole expectation on
        // the pair's first publish) and going dark when prefill ends. The
        // in-flight rate is the block gauge's movement over the span since
        // it last moved.
        let trace = [
            Scrape {
                running: true,
                pre_total: 400000.0,
                prefilling: true,
                ..base
            },
            Scrape {
                running: true,
                expected: 1075.0,
                pre_total: 400300.0,
                prefilling: true,
                ..base
            },
            Scrape {
                running: true,
                pre_live: 1075.0,
                expected: 1075.0,
                pre_total: 400575.0,
                prefilling: true,
                ..base
            },
            // a dark flutter of the publish cadence: the pair publishes
            // nothing this scrape
            Scrape {
                running: true,
                pre_total: 401075.0,
                prefilling: true,
                ..base
            },
            // prefill done: the expectation clears, the live gauge holds one
            // publish (the engine's shape), the decode gauge moves
            Scrape {
                running: true,
                gen_live: 81470.0,
                pre_total: 401075.0,
                prefilling: false,
                ..base
            },
            Scrape {
                running: true,
                gen_live: 81593.0,
                pre_total: 401075.0,
                prefilling: false,
                ..base
            },
            Scrape {
                running: true,
                gen_live: 81751.0,
                pre_total: 401075.0,
                prefilling: false,
                ..base
            },
            // completion — and the successor's prefill lit on the same
            // scrape: continuous batching publishes the successor's pair
            // alongside the old request's completion counters. The old
            // window's rate is gone with its request (perf.rs's measured-
            // TTFT path owns the finished request's exact mean now); the
            // successor's window opens on this scrape.
            Scrape {
                gen_total: 81833.0,
                succeeded: 301.0,
                running: true,
                gen_live: 81833.0,
                pre_live: 0.0,
                expected: 900.0,
                pre_total: 401275.0,
                prefilling: true,
            },
            Scrape {
                gen_total: 81833.0,
                succeeded: 301.0,
                running: true,
                gen_live: 81833.0,
                pre_live: 900.0,
                expected: 900.0,
                pre_total: 401575.0,
                prefilling: true,
            },
            Scrape {
                gen_total: 81833.0,
                succeeded: 301.0,
                running: true,
                gen_live: 81833.0,
                pre_live: 0.0,
                expected: 0.0,
                pre_total: 401975.0,
                prefilling: true,
            },
        ];
        let mut obs = Vec::new();
        for (i, sc) in trace.iter().enumerate() {
            let (mut s, _) = adapter.observe(&counters(*sc));
            lr.observe(&counters(*sc), &live(*sc), &mut s);
            assert!(s.processing, "poll {i}");
            obs.push(s);
            // The engine's shape: the expectation is the prompt's exact
            // size and arrives at the prefill's first publish; the live
            // gauge
            // is the honest progress line before it. The panels hold that
            // line through the cadence's dark flutters, but nothing may run
            // away past the request's own published prompt — the runaway
            // the parked adapter showed bled stale totals across the race
            // without bound.
            // The panels show the engine's latest publication: the open
            // request's prompt, never more, never a stale larger one.
            let bound = if i >= 7 { 900 } else { 1075 };
            assert!(obs[i].prompt_processed <= bound, "poll {i}");
            if sc.expected > 0.0 {
                // The expectation landing is the prompt's exact size — the
                // panels must show it whole on that very scrape.
                assert_eq!(obs[i].prompt_processed, bound, "poll {i}");
            }
        }
        // The progress line before the block gauge publishes is the phase
        // pin's floor: one processed token (perf.rs infers the phase from
        // movement; a floor of one holds Prefill while the metrics' block
        // publishing keeps the counters frozen).
        assert_eq!(
            obs[0].prompt_processed, 1,
            "the phase pin's one-token floor"
        );
        assert_eq!(obs[1].prompt_processed, 1075, "the expectation, whole");
        // Decode polls ride the live decode gauge while the completion
        // counters sit frozen — the whole point of the live path.
        assert_eq!(obs[4].decoded, 18);
        assert_eq!(obs[5].decoded, 141, "81593 - 81452");
        assert_eq!(obs[6].decoded, 299);
        // The in-flight rate is the step counter's movement over the span
        // since the window opened — a span mean, charged every scrape of
        // the open window (lit publishes and dark flutters alike) and never
        // divided by the expectation's own landing. It runs through the
        // request's decode phase and ends with the request: obs[7]'s
        // completion scrape hands the finished request's exact whole-span
        // rate to perf.rs's measured-TTFT path.
        // The engine publishes a rate: the step counter's movement since
        // the scrape the last rate published, over the span since then. The
        // expectation's whole-size landing is a value, never a division
        // input (the base anchors at the landing scrape's own counter
        // value); the first reading is the next interval's 275 tokens over
        // 0.1 s.
        assert_eq!(obs[0].prefill_tps, None, "the window has not moved yet");
        assert_eq!(
            obs[1].prefill_tps, None,
            "the expectation lands whole — a value, not a rate"
        );
        let rate = obs[2].prefill_tps.expect("the first published interval");
        assert!((rate - 2750.0).abs() < 1.0, "275 / 0.1 s, got {rate}");
        let rate = obs[3].prefill_tps.expect("the second published interval");
        assert!((rate - 5000.0).abs() < 1.0, "500 / 0.1 s, got {rate}");
        // Decode: the step counter holds its value — nothing new to publish,
        // and the last reading holds with it until the request retires: the
        // panel's big numeral never blinks mid-request. The reading retires
        // with the request: obs[7]'s completion scrape hands the finished
        // request's exact whole-span mean to perf.rs's measured-TTFT path.
        let rate = obs[4].prefill_tps.expect("decode holds the last reading");
        assert!((rate - 5000.0).abs() < 1.0, "holds, got {rate}");
        let rate = obs[6].prefill_tps.expect("still decoding: holds");
        assert!((rate - 5000.0).abs() < 1.0, "holds, got {rate}");
        // The completion scrape: the adapter's exact counters pass through,
        // and the successor's expectation opens its own window on this very
        // scrape — keyed to its own size, anchoring its base at its landing.
        // The old request's reading is gone with it (perf.rs owns the
        // finished request's exact mean).
        assert_eq!(obs[7].decoded, 381, "81833 - 81452");
        assert_eq!(
            obs[7].prompt_processed, 900,
            "the successor's prompt lands with its expectation"
        );
        assert_eq!(
            obs[7].prefill_tps, None,
            "the reading retires with the request — perf.rs owns the exact mean"
        );
        assert_eq!(
            obs[8].prompt_processed, 900,
            "held through the successor's own dark flutter"
        );
        // The successor's own window: the landing scrape contributes no
        // reading (the interval ending at it is not fully charged); the
        // next scrape's 300 tokens divide one whole 0.1 s interval, and
        // the 400 after that repeats the honest per-interval rate.
        let rate = obs[8].prefill_tps.expect("the successor's first interval");
        assert!((rate - 3000.0).abs() < 0.01, "300 / 0.1 s: {rate}");
        let rate = obs[9].prefill_tps.expect("the successor's second interval");
        assert!((rate - 4000.0).abs() < 0.01, "400 / 0.1 s: {rate}");
    }

    #[test]
    fn completion_with_a_warm_gauge_passes_through() {
        let mut lr = LiveRates::default();
        let mut adapter = crate::vllm::VllmAdapter::new();
        let base = Scrape {
            gen_total: 81452.0,
            succeeded: 300.0,
            running: false,
            gen_live: 81452.0,
            pre_live: 0.0,
            expected: 0.0,
            pre_total: 400000.0,
            prefilling: false,
        };
        for _ in 0..2 {
            let (mut s, _) = adapter.observe(&counters(base));
            lr.observe(&counters(base), &live(base), &mut s);
        }
        let done = Scrape {
            gen_total: 82527.0,
            succeeded: 301.0,
            running: false,
            gen_live: 82527.0,
            pre_live: 0.0,
            expected: 0.0,
            pre_total: 401075.0,
            prefilling: false,
        };
        // The request's prefill lit and closed between the two idle primes;
        // the completion scrape finds everything dark.
        let lit_scrapes = [
            Scrape {
                running: true,
                pre_total: 400400.0,
                prefilling: true,
                ..base
            },
            Scrape {
                running: true,
                pre_live: 1075.0,
                expected: 1075.0,
                pre_total: 401075.0,
                prefilling: true,
                ..base
            },
        ];
        for sc in lit_scrapes {
            let (mut s, _) = adapter.observe(&counters(sc));
            lr.observe(&counters(sc), &live(sc), &mut s);
        }
        let (mut s, _) = adapter.observe(&counters(done));
        lr.observe(&counters(done), &live(done), &mut s);
        assert!(!s.processing);
        assert_eq!(s.decoded, 1075, "the exact completion counter");
        // The prompt the panels show at completion: the adapter's own
        // `vllm:prompt_tokens_total` movement (which mlx-serve publishes at
        // completion — `counters()` carries `pre_total` as it). The engine's
        // live publication retires with the request.
        assert_eq!(
            s.prompt_processed, 1075,
            "the exact completion counter, as the adapter sees it"
        );
    }

    #[test]
    fn mtp_identity_survives_the_counters_being_blank() {
        // mlx-serve says `mtp_loaded` on /v1/models but publishes no specu-
        // lative counters at all: the shared adapter's answer is "none",
        // and the panel would read it as "the server is not drafting". The
        // identity's word wins.
        let mut lr = live_rates(true);
        let mut adapter = crate::vllm::VllmAdapter::new();
        let base = Scrape {
            gen_total: 1000.0,
            succeeded: 10.0,
            running: true,
            gen_live: 1000.0,
            pre_live: 0.0,
            expected: 0.0,
            pre_total: 5000.0,
            prefilling: false,
        };
        let (mut s, _) = adapter.observe(&counters(base));
        lr.observe(&counters(base), &live(base), &mut s);
        assert_eq!(
            s.spec_types, "mtp",
            "the identity's answer, not the adapter's"
        );
    }

    /// The engine's prefill counter (`prefill_tokens_total`) is the one value
    /// it publishes exactly, per step, every scrape — the live pair lags a
    /// beat and flutters. So the in-flight rate is its movement over the
    /// samples the engine's own signal kept lit: it never reads zero while a
    /// prefill runs, the expectation's step (the prompt's exact size, landing
    /// at the prefill's first publish) never causes a flood, and the
    /// request's
    /// whole-span rate lands when the signal closes.
    #[test]
    fn prefill_rate_tracks_the_engine_counter_and_lands_at_the_close() {
        let mut lr = LiveRates::default();
        let mut adapter = crate::vllm::VllmAdapter::new();
        let base = Scrape {
            gen_total: 1000.0,
            running: false,
            succeeded: 10.0,
            gen_live: 1000.0,
            pre_live: 0.0,
            expected: 0.0,
            pre_total: 5000.0,
            prefilling: false,
        };
        let (mut s, _) = adapter.observe(&counters(base));
        lr.observe(&counters(base), &live(base), &mut s);
        let trace = [
            // the request is admitted and its prefill begins: the phase
            // bit lights and the expectation — the prompt's exact size —
            // lands whole at the prefill's first publish. The step counter
            // moves 400 on the very scrape the expectation arrives.
            Scrape {
                running: true,
                expected: 1600.0,
                pre_total: 5400.0,
                prefilling: true,
                ..base
            },
            // the step counter moves every scrape the engine prefills
            Scrape {
                running: true,
                expected: 1600.0,
                pre_total: 5800.0,
                prefilling: true,
                ..base
            },
            // a dark flutter of the publish cadence: the pair publishes
            // nothing, the step counter keeps moving, the span keeps
            // charging
            Scrape {
                running: true,
                expected: 1600.0,
                pre_total: 6200.0,
                prefilling: true,
                ..base
            },
            // the block gauge publishes the prefill's completion as the
            // prefill closes; the step counter has reached the prompt whole
            Scrape {
                running: true,
                pre_live: 0.0,
                expected: 1600.0,
                pre_total: 6600.0,
                prefilling: true,
                ..base
            },
            // decode: the step counter holds its value, the pair goes dark
            Scrape {
                running: true,
                gen_live: 1005.0,
                pre_total: 6600.0,
                prefilling: false,
                ..base
            },
            // completion — and the successor's expectation already lit on
            // the same scrape (continuous batching shares one publish block
            // per metric between the closing request's counters and the
            // successor's prefill)
            Scrape {
                gen_total: 1008.0,
                succeeded: 11.0,
                running: true,
                gen_live: 1008.0,
                pre_live: 0.0,
                expected: 900.0,
                pre_total: 6600.0,
                prefilling: true,
            },
            // the successor's own prefill moving the step counter
            Scrape {
                gen_total: 1008.0,
                succeeded: 11.0,
                running: true,
                gen_live: 1008.0,
                pre_live: 0.0,
                expected: 900.0,
                pre_total: 7000.0,
                prefilling: true,
            },
        ];
        let mut obs = Vec::new();
        for sc in trace.iter() {
            let (mut s, _) = adapter.observe(&counters(*sc));
            lr.observe(&counters(*sc), &live(*sc), &mut s);
            obs.push(s);
        }
        // In-flight rate from the engine's per-step counter: the movement
        // since the last publish over the span since the last publish — the
        // engine's own step-by-step rate, not a diluting whole-span mean.
        // The expectation's whole-size landing is a value, never a division
        // input (the base swallows it on the landing scrape); the publish
        // cadence's dark flutters move the step counter too — the engine
        // prefills while its paired gauges blink — and a reading waits for
        // a fully charged interval, so it lands one scrape after the window
        // opens and every scrape after that over one 0.1 s interval. Once
        // the counter holds its value the reading holds with it: the panel's
        // big numeral shows the last published rate through the dark beats
        // rather than blinking.
        assert_eq!(
            obs[0].prefill_tps, None,
            "the expectation lands whole — a value, not a rate"
        );
        assert_eq!(
            obs[1].prefill_tps, None,
            "the interval ending here is not fully charged yet"
        );
        let rate = obs[2].prefill_tps.expect("the window's first reading");
        assert!((rate - 4000.0).abs() < 0.01, "400 / 0.1 s: {rate}");
        let rate = obs[3]
            .prefill_tps
            .expect("the prefill's last step republishes");
        assert!((rate - 4000.0).abs() < 0.01, "400 / 0.1 s, got {rate}");
        // Decode: the step counter holds its value — no new reading, and
        // the last one holds until the request retires.
        let rate = obs[4].prefill_tps.expect("decode holds the last reading");
        assert!((rate - 4000.0).abs() < 0.01, "holds 400 / 0.1 s: {rate}");
        // The prompt's computed total appears whole when the expectation
        // lands and holds through the decode scrape; the whole-span rate is
        // the completion counters' job (perf.rs's measured-TTFT path).
        assert_eq!(obs[0].prompt_processed, 1600, "the expectation, whole");
        assert_eq!(obs[4].prompt_processed, 1600, "held through decode");
        assert_eq!(obs[4].decoded, 5, "the decode gauge already moved");
        // The completion scrape: perf.rs's exact counters take the panel
        // back — the finished request's live reading retires — and the
        // successor's expectation opens its own window right here.
        assert_eq!(obs[5].decoded, 8, "1008 - 1000");
        assert_eq!(
            obs[5].prompt_processed, 900,
            "the successor's prompt lands with its expectation"
        );
        assert_eq!(
            obs[5].prefill_tps, None,
            "the finished request's span is closed; the successor's has no movement yet"
        );
        // The successor's own window, measuring only its own movement: its
        // landing scrape contributes nothing (the base anchors at it), the
        // next scrape's 400 tokens divide one whole 0.1 s interval.
        let rate = obs[6]
            .prefill_tps
            .expect("the successor's second scrape measures");
        assert!((rate - 4000.0).abs() < 0.01, "400 / 0.1 s: {rate}");
    }

    #[test]
    fn counter_restart_reanchors_without_negative_counts() {
        let mut lr = LiveRates::default();
        let mut adapter = crate::vllm::VllmAdapter::new();
        let big = Scrape {
            gen_total: 5000.0,
            running: true,
            succeeded: 50.0,
            gen_live: 5060.0,
            pre_live: 0.0,
            expected: 0.0,
            pre_total: 90000.0,
            prefilling: false,
        };
        for _ in 0..2 {
            let (mut s, _) = adapter.observe(&counters(big));
            lr.observe(&counters(big), &live(big), &mut s);
        }
        // The server restarts: gauges and counters rewind together.
        let small = Scrape {
            gen_total: 10.0,
            running: true,
            succeeded: 0.0,
            gen_live: 4.0,
            pre_live: 0.0,
            expected: 0.0,
            pre_total: 20.0,
            prefilling: false,
        };
        let (mut s, _) = adapter.observe(&counters(small));
        lr.observe(&counters(small), &live(small), &mut s);
        assert!(s.decoded < 1000, "no phantom token flood from the rewind");
    }

    #[test]
    fn a_long_prefill_publishes_steps_and_measured_rates_all_the_way_up() {
        // Shape captured live from the M5 Max server during a 29k-token
        // prefill (~16 s): the expectation lands whole at the prefill's
        // first publish, `prefill_tokens_live` steps 8192 → 16384 → 24576
        // while `requests_prefilling` stays lit the whole span, and
        // `prefill_tokens_total` moves ~4k tokens every scrape. The old
        // model — expectation arriving "at the prefill's last step" — came
        // from scrapes that caught completion polls; this is the truth.
        let mut lr = LiveRates::default();
        let mut adapter = crate::vllm::VllmAdapter::new();
        let base = Scrape {
            gen_total: 1000.0,
            running: false,
            succeeded: 10.0,
            gen_live: 1000.0,
            pre_live: 0.0,
            expected: 0.0,
            pre_total: 5000.0,
            prefilling: false,
        };
        let (mut s, _) = adapter.observe(&counters(base));
        lr.observe(&counters(base), &live(base), &mut s);
        let trace = [
            // expectation lit, nothing moved yet — the prefill has begun
            Scrape {
                running: true,
                expected: 28867.0,
                pre_total: 5000.0,
                prefilling: true,
                ..base
            },
            // the engine's step counter moves while `live` still reads 0
            Scrape {
                running: true,
                expected: 28867.0,
                pre_total: 9000.0,
                prefilling: true,
                ..base
            },
            // the step gauges publish their first chunk
            Scrape {
                running: true,
                pre_live: 8192.0,
                expected: 28867.0,
                pre_total: 13000.0,
                prefilling: true,
                ..base
            },
            Scrape {
                running: true,
                pre_live: 16384.0,
                expected: 28867.0,
                pre_total: 17000.0,
                prefilling: true,
                ..base
            },
            Scrape {
                running: true,
                pre_live: 24576.0,
                expected: 28867.0,
                pre_total: 21000.0,
                prefilling: true,
                ..base
            },
            // prefill done: gauges dark, decode has started
            Scrape {
                running: true,
                gen_live: 1003.0,
                prefilling: false,
                ..base
            },
        ];
        let mut obs = Vec::new();
        for sc in trace {
            let (mut s, _) = adapter.observe(&counters(sc));
            lr.observe(&counters(sc), &live(sc), &mut s);
            obs.push(s);
        }
        // The prompt the panels show tracks the published steps from the
        // first lit scrape, and never decreases as the prefill runs.
        for (i, o) in obs[..5].iter().enumerate() {
            assert_eq!(
                o.prompt_processed, 28867,
                "the expectation lands whole and holds (obs {i})"
            );
        }
        // A measured rate exists at every scrape after the window's opening
        // beat: each reading is the movement since the last publish over
        // the span since the last publish — the engine's per-interval rate,
        // one scrape behind the counter it tracks. The expectation's land-
        // ing opens the window and is itself a value, not a measurement —
        // never divided — and the first reading waits for the next scrape's
        // whole interval: 4000 / 0.1 s = 40000 tokens/s, the engine's real
        // rate, holding step-for-step all the way to the prefill's close.
        // The rate stays alive through the request's decode phase (nothing
        // moves, nothing publishes, the reading holds); completion's exact
        // counters, via perf.rs's measured-TTFT path, own the number from
        // the close on.
        assert_eq!(
            obs[0].prefill_tps, None,
            "the expectation lands whole — a value, not a rate"
        );
        assert_eq!(
            obs[1].prefill_tps, None,
            "the interval ending here is not fully charged yet"
        );
        assert!(
            obs[2].prefill_rate_live && obs[4].prefill_rate_live,
            "in-flight readings are tagged live for the panel"
        );
        let r2 = obs[2].prefill_tps.expect("the window's first reading");
        assert!((r2 - 40000.0).abs() < 0.01, "4000 / 0.1 s: {r2}");
        let r3 = obs[3].prefill_tps.expect("second publish is a rate");
        assert!((r3 - 40000.0).abs() < 0.01, "4000 / 0.1 s: {r3}");
        let r4 = obs[4].prefill_tps.expect("rate all the way up");
        assert!((r4 - 40000.0).abs() < 0.01, "4000 / 0.1 s: {r4}");
        // The final scrape closes the request (succeeded moves): the span
        // mean is retired with it — perf.rs's measured-TTFT path reports
        // the request's exact whole-span rate from here on.
        assert_eq!(obs[5].prefill_tps, None);
        assert!(!obs[5].prefill_rate_live);
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    /// Point MLX_METRICS_PORT at a running mlx-serve to prove the whole v1
    /// path — identity, the metrics scrape, and the shared vLLM adapter —
    /// against a live engine.
    #[tokio::test]
    #[ignore = "requires a running mlx-serve and MLX_METRICS_PORT"]
    async fn live_endpoint() {
        let port = std::env::var("MLX_METRICS_PORT").unwrap().parse().unwrap();
        let auth = HttpAuth::from_token(std::env::var("MLX_METRICS_KEY").ok());
        let served = poll_served_model("127.0.0.1", port, &auth)
            .await
            .expect("served id from /v1/models");
        let (name, ctx) = (served.id, served.ctx);
        assert!(ctx > 0, "the served entry carries its context length");
        let (counters, live) = poll("127.0.0.1", port, &name, &[], &auth)
            .await
            .expect("vllm-named counters scrape");
        assert!(counters.generation_total > 0.0);
        assert!(counters.ttft_count > 0.0);
        let (stats, _spec) = crate::vllm::VllmAdapter::new().observe(&counters);
        assert!(stats.ctx_max == 0 || stats.ctx_max > 1);
        let mut lr = live_rates(served.mtp);
        let mut stats = stats;
        lr.observe(&counters, &live, &mut stats);
        assert_eq!(
            stats.spec_types, "mtp",
            "the engine says MTP is loaded, the panel tag follows"
        );
        eprintln!(
            "mlx live: name={name} ctx={ctx} gen_total={} live_gen={} mtp={} ok",
            counters.generation_total, live.generation_live, served.mtp
        );
    }
}
