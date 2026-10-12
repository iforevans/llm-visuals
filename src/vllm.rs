//! vLLM metrics adapter.
//!
//! llama.cpp's `/slots` endpoint reports per-request counters — `n_decoded`
//! and friends are reset each time a slot takes a new task — exactly the
//! shape the `perf.rs` state machine consumes. vLLM's Prometheus endpoint
//! only exposes engine-wide cumulative counters, so this module
//! reconstructs the per-request view on top of them:
//!
//! * a baseline is captured at the moment a request is admitted; a
//!   per-request value is `counter - baseline`;
//! * a synthetic task id advances when vLLM reports a completion
//!   (`vllm:request_success_total`), which is what `perf::observe` keys on.
//!
//! The boundary signal is the success counter rather than
//! `num_requests_running`: continuous batching admits the next request in
//! the same step the previous one finishes, so the running gauge never dips
//! to zero between chained requests. When the successor is admitted within
//! the same poll window as the completion, the baseline re-anchors to the
//! current sample (mirroring llama.cpp's wholesale re-read of the counters
//! on a new task) — the finishing request's tail tokens in that window are
//! dropped rather than double-counted. When requests are separated by an
//! idle gap the baseline re-anchors to the previous sample instead, so the
//! closing sample still carries the full request.
//!
//! A `model_name` label guard rejects samples served by a different model:
//! a containerized vLLM instance published to a port another local server
//! already owns would otherwise feed this slot the other model's counters
//! (both "listen" on the same port from the host's point of view).
//! With the guard such a slot's samples are rejected and the poll loop
//! decays it to idle instead of lying.

use crate::observe::{http_get, ClosingRequest, HttpAuth, LiveStats, SpecMetrics};

/// Engine-wide vLLM counters (summed across engines/replicas) as scraped
/// from `/metrics`.
#[derive(Debug, Clone, Default)]
pub struct VllmCounters {
    pub prompt_total: f64,
    pub generation_total: f64,
    pub cached_total: f64,
    pub running: f64,
    pub succeeded: f64,
    /// vllm:time_to_first_token_seconds_sum{...}
    pub ttft_sum: f64,
    /// vllm:time_to_first_token_seconds_count{...}
    pub ttft_count: f64,
    /// vllm:inter_token_latency_seconds_sum{...}
    pub itl_sum: f64,
    /// vllm:request_prompt_tokens_sum/count{...} and
    /// vllm:request_generation_tokens_sum/count{...}: per-request sums.
    /// A count delta of 1 means the sum delta is exactly one request's
    /// prompt (or generation) size, unshared with any straddling request.
    pub req_prompt_sum: f64,
    pub req_prompt_count: f64,
    pub req_gen_sum: f64,
    pub req_gen_count: f64,
    pub spec_drafts: f64,
    pub spec_draft_tokens: f64,
    pub spec_accepted: f64,
    /// Speculative positions that drafted at least one token; its length is
    /// the MTP depth shown by the panel (base-0 or base-1 label schemes both
    /// count the same).
    pub spec_positions: Vec<u32>,
    /// The speculative backend when the server names it (NInfer's
    /// `speculative_backend` label); the panel's title. None reads as MTP.
    pub spec_backend: Option<String>,
    /// The context limit when the server reports one (NInfer's
    /// `ninfer_max_context_tokens`); 0 when it does not.
    pub max_context: usize,
    /// Served model name from the metrics labels; the identity marker for
    /// the cross-wiring guard.
    pub model_name: Option<String>,
}

/// Parse the Prometheus text exposition for the engine-wide counters we
/// need. `None` when the body carries no `vllm:` token counters (e.g. the
/// 404 page a llama.cpp server answers `/metrics` with).
pub fn parse_vllm_metrics(body: &str) -> Option<VllmCounters> {
    let mut c = VllmCounters::default();
    let mut saw_tokens = false;

    for line in body.lines() {
        let Some(rest) = line.strip_prefix("vllm:") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let Some(head) = it.next() else {
            continue;
        };
        let Some(val) = it.next().and_then(|v| v.parse::<f64>().ok()) else {
            continue;
        };
        // vLLM glues the label block to the metric name
        // (`prompt_tokens_total{engine="0",...}`); cut at the first `{`.
        let (name, labels) = match head.find('{') {
            Some(i) => (&head[..i], Some(head[i + 1..].trim_end_matches('}'))),
            None => (head, None),
        };

        match name {
            "prompt_tokens_total" => {
                c.prompt_total += val;
                saw_tokens = true;
            }
            "generation_tokens_total" => {
                c.generation_total += val;
                saw_tokens = true;
            }
            "prompt_tokens_cached_total" => c.cached_total += val,
            "num_requests_running" => c.running = c.running.max(val),
            "request_success_total" => c.succeeded += val,
            "time_to_first_token_seconds_sum" => c.ttft_sum += val,
            "time_to_first_token_seconds_count" => c.ttft_count += val,
            "inter_token_latency_seconds_sum" => c.itl_sum += val,
            // Per-request histograms: the exact token counts of the
            // requests that completed since the previous scrape. Unlike
            // the running totals these cannot straddle two requests, so
            // they correct the adapter's whole-counter spans when the
            // completion edge lands over a busier window than one request.
            "request_prompt_tokens_sum" => c.req_prompt_sum += val,
            "request_prompt_tokens_count" => c.req_prompt_count += val,
            "request_generation_tokens_sum" => c.req_gen_sum += val,
            "request_generation_tokens_count" => c.req_gen_count += val,
            "spec_decode_num_drafts_total" => c.spec_drafts += val,
            "spec_decode_num_draft_tokens_total" => c.spec_draft_tokens += val,
            "spec_decode_num_accepted_tokens_total" => c.spec_accepted += val,
            "spec_decode_num_accepted_tokens_per_pos_total" => {
                if val > 0.0 {
                    if let Some(p) = labels
                        .and_then(|l| label_value(l, "position"))
                        .and_then(|p| p.parse::<u32>().ok())
                    {
                        if !c.spec_positions.contains(&p) {
                            c.spec_positions.push(p);
                        }
                    }
                }
            }
            _ => {}
        }

        if c.model_name.is_none() {
            c.model_name = labels.and_then(|l| label_value(l, "model_name"));
        }
    }

    saw_tokens.then_some(c)
}

/// Pull one label out of a `{k1="v1", k2="v2"}` set.
fn label_value(set: &str, key: &str) -> Option<String> {
    for kv in set.split(',') {
        let kv = kv.trim();
        if let Some(v) = kv.strip_prefix(key) {
            if let Some(v) = v.strip_prefix('=') {
                return Some(v.trim_matches('"').to_string());
            }
        }
    }
    None
}

/// GET `/metrics` on `port` and guard the result against cross-wiring. The
/// expected name is the alias (`--served-model-name`) or the model file's
/// stem; a mismatch is fatal only when the label clearly belongs to another
/// detected model listening on the same port — the signature of a
/// container port-publish collision, where accepting would silently show
/// the other model's counters.
pub async fn poll_vllm(
    host: &str,
    port: u16,
    expected_name: &str,
    other_models: &[(String, u16)],
    auth: &HttpAuth,
) -> Option<VllmCounters> {
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
    let c = parse_vllm_metrics(&body)?;
    if c.model_name
        .as_deref()
        .is_some_and(|label| cross_wired(label, expected_name, port, other_models))
    {
        return None;
    }
    Some(c)
}

/// Whether a `model_name` label that disagrees with the expected name is
/// evidence of a container port-publish collision: another detected model
/// claims the same port and the label matches *its* name.
pub(crate) fn cross_wired(
    label: &str,
    expected: &str,
    port: u16,
    others: &[(String, u16)],
) -> bool {
    label != expected
        && others.iter().any(|(name, p)| {
            *p == port
                && name != expected
                && name.len() >= 4
                && (label == name || label.starts_with(name.as_str()) || name.starts_with(label))
        })
}

/// Per-slot reconstruction state: counter baselines + the synthetic task
/// id advanced on completions.
#[derive(Debug, Default)]
pub struct VllmAdapter {
    primed: bool,
    prev_prompt: f64,
    prev_generation: f64,
    prev_cached: f64,
    prev_succeeded: f64,
    /// `succeeded` at the point the current request's window opened: the
    /// idle-edge close's whole-token spans span base_prompt..prompt_total,
    /// and the completions that moved that ledger over the same span are
    /// `succeeded - base_succeeded`. Opened with the whole-request
    /// baselines (admission or the previous completion's re-anchor).
    base_succeeded: f64,
    prev_running: bool,
    prev_ttft_sum: f64,
    prev_ttft_count: f64,
    prev_itl_sum: f64,
    base_prompt: f64,
    base_generation: f64,
    base_cached: f64,
    base_ttft_sum: f64,
    base_ttft_count: f64,
    base_itl_sum: f64,
    /// The whole-request anchors in effect when the finished request was
    /// admitted, snap-shotted at the top of every scrape before any branch
    /// below moves them (the fresh-admission re-anchor or a completion's
    /// re-anchor): the edge scrape's recovery below still names the finished
    /// request through them.
    edge_base_prompt: f64,
    edge_base_cached: f64,
    edge_base_generation: f64,
    edge_base_ttft_sum: f64,
    edge_base_ttft_count: f64,
    edge_base_itl_sum: f64,
    task_seq: i64,
    /// Set on the poll where a completion and a successor's admission
    /// share one scrape; consumed into LiveStats.closing on that poll.
    pending_close: Option<ClosingRequest>,
    /// Mlx's whole-request prefill mean, staged by `LiveRates` on the
    /// scrape the engine's vllm-named counters moved — the engine's com-
    /// pletion, which the adapter's `running` phase may read lit for one
    /// more poll. Held until a closing view is built.
    pending_measured_rate: Option<f32>,
    pending_measured_decode: Option<f32>,
    /// The exact span behind `pending_measured_rate` (tokens ÷ rate): the
    /// engine's measured prefill duration, which *is* the request's time-
    /// to-first-token. Survives into `ClosingRequest.prefill_secs`.
    pending_prefill_secs: Option<f64>,
    /// The finished request's spans, captured on the scrape the success
    /// edge saw. Mlx's completion straddles two scrapes: the counters
    /// move while `running` stays lit, and the fresh-admission re-anchoring
    /// on the successor's scrape would erase them. The edge hands the
    /// panels the staged close instead of rebuilding one from moved
    /// baselines.
    staged_close: Option<ClosingRequest>,
    /// The open request's expected prompt size, taken from mlx's publish
    /// at admission and held across the frozen counters: a completion
    /// scrape that sees no token movement at all (mlx's counters can move
    /// twice between two adapter polls under churn) still names the
    /// finished request's prompt through this. 0 when the server publishes
    /// no live gauges (plain vLLM), where the whole-counter capture is
    /// already exact and must not be overwritten.
    pending_prompt: usize,
    /// The open request's generation size as it decodes — the live gauge's
    /// running total while the request is lit. Consumed with `pending_prompt`
    /// by the blank-blank fallback.
    pending_gen: usize,
    /// Whether `pending_prompt`/`pending_gen` describe the request that is
    /// *in flight right now* (a note from a lit scrape) or a completion's
    /// residue (a note from a dark scrape). The blank-blank fallback only
    /// trusts an in-flight memory: under continuous batching the next
    /// request is already lit when the previous one's close is built, and
    /// a stale completion's sizes would print someone else's prompt.
    pending_in_flight: bool,
    /// Whether mlx's live gauges published the in-flight request's exact
    /// prompt while it ran (see `note_measured_prompt`).
    pending_span_known: bool,
    /// The whole prompt of the request currently in flight, as published
    /// by the engine's live gauges (fresh expectation + cache-restored
    /// live progress). Stamped on the close that names this request.
    pending_span_size: u64,
    /// The whole-token counters' position as of the scrape *before* the
    /// in-flight request's admission (the last scrape that predates the
    /// adapter's baseline window for it). A completion that lands while
    /// the request is in flight — under continuous batching the engine
    /// finishes foreign requests around the tracked one — moves the
    /// per-request histograms by exactly the foreign request's sizes:
    /// the sums over this anchor, with a count delta of one, name that
    /// foreign completion. Crediting it to the in-flight record is what
    /// printed the churner's 2,805-token prompt on the probe's row; the
    /// sums over the *unmoved* anchor are the tracked request's own.
    pre_window_req_sum: f64,
    pre_window_gen_sum: f64,
    pre_window_req_count: f64,
    pre_window_gen_count: f64,
    /// `request_prompt_tokens_count - prompt_tokens_total` as of the last
    /// scrape, and the same for generation: the two ledgers move together
    /// (the whole-counter carries the request's exact prompt at
    /// completion), so a moved offset proves the counters moved on scrapes
    /// the adapter never saw — a close built from them names no request.
    /// Survives the completion-edge re-anchoring: the offset is a property
    /// of the server's bookkeeping, not of any one request's window.
    last_hist_offset: f64,
    last_hist_gen_offset: f64,
    offset_seen: bool,
    /// Per-request histogram positions (see `VllmCounters`): the exact
    /// token counts of the requests that completed since a given scrape.
    /// A completion window whose movement outruns `prompt / ttft` — the
    /// biggest rate the counters can honestly produce — straddled more
    /// than the finished request; these histograms name its true sizes.
    prev_req_sum: f64,
    prev_gen_sum: f64,
    prev_req_count: f64,
    prev_gen_count: f64,
    /// The histograms' positions as of the last completion the adapter
    /// observed. The sums over these anchors, with a count delta of
    /// exactly one, are the in-flight request's exact sizes — `last_*`
    /// because mlx credits the histograms while `running` stays lit, so
    /// the whole-request baselines (`base_*`) move past them before the
    /// phase edge ever sees the completion.
    last_req_sum: f64,
    last_gen_sum: f64,
    last_req_count: f64,
    last_gen_count: f64,
}

impl VllmAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage engine-published whole-request means (mlx-serve's `LiveRates`
    /// calls this the scrape its window closes) so the next closing view
    /// this adapter builds carries them, whatever race hid the counters.
    /// `request_prompt` is the in-flight request's exact prompt size from
    /// the live gauges: a close that no counter position can bracket takes
    /// it as its prompt. It is noted afresh on every scrape that publishes
    /// one — the adapter clears it when it consumes a close, so a stale
    /// expectation can never masquerade as a later request's size.
    pub fn note_measured_rates(
        &mut self,
        prefill: Option<f32>,
        decode: Option<f32>,
        prefill_secs: Option<f64>,
        request_prompt: usize,
        request_gen: usize,
    ) {
        // Preserve-on-None: mlx stages a rate on the scrapes its window
        // model publishes one and calls here at completion; a note with no
        // rate means "nothing new staged", not "forget the staged pair" —
        // the pair is staged from the window's snapshot and stays true
        // until a close consumes it. (vLLM's own rate notes, when they
        // come, are Some and overwrite.)
        if prefill.is_some() {
            self.pending_measured_rate = prefill;
        }
        if decode.is_some() {
            self.pending_measured_decode = decode;
        }
        if prefill_secs.is_some() {
            self.pending_prefill_secs = prefill_secs;
        }
        if request_prompt > 0 {
            self.pending_prompt = request_prompt;
        }
        if request_gen > 0 {
            self.pending_gen = request_gen;
        }
        // A rates-note always arrives with (or just after) a completion:
        // its sizes describe the request that *finished*, not whoever is
        // lit now. The blank-blank fallback must not read them for the
        // in-flight request's span.
        self.pending_in_flight = false;
    }

    /// Note the in-flight request's own published sizes (mlx's live gauges:
    /// the prompt's exact size at admission, the generation total as it
    /// decodes). These describe the request the panels are watching *while
    /// it runs*, so the blank-blank fallback may take them as its span.
    /// `span_known` says the engine published the request's exact prompt
    /// while it ran: a completion close that straddled foreign counter
    /// movement keeps the published span and hands over only its timing.
    pub fn note_measured_prompt(
        &mut self,
        request_prompt: usize,
        request_gen: usize,
        span_known: bool,
    ) {
        if request_prompt > 0 {
            self.pending_prompt = request_prompt;
        }
        if request_gen > 0 {
            self.pending_gen = request_gen;
        }
        self.pending_in_flight = true;
        self.pending_span_known = span_known;
    }

    /// The whole prompt (fresh expectation + cache-restored live progress)
    /// the engine's live gauges publish for the request currently in
    /// flight. Stamped on the next close the adapter builds for it, giving
    /// perf a span the completion's whole-counter ledger cannot outrun
    /// without having absorbed a foreign publication.
    pub fn note_span_size(&mut self, whole_prompt: u64) {
        if whole_prompt == 0 {
            return;
        }
        if self.pending_span_size > 0 && whole_prompt != self.pending_span_size {
            // a different span is a different request — the engine names
            // each request's prompt whole at admission — so the old span
            // belonged to a request that closed unseen, and the close that
            // names *this* one clamps to this value.
            self.pending_span_size = whole_prompt;
        } else if self.pending_span_size == 0 {
            self.pending_span_size = whole_prompt;
        }
    }

    /// Advance the state machine with a fresh `/metrics` scrape. The
    /// caller sets `stats.ctx_max` (the adapter doesn't know it).
    pub fn observe(&mut self, c: &VllmCounters) -> (LiveStats, Option<SpecMetrics>) {
        let running = c.running > 0.5;

        // The per-request histograms move only when a request completes,
        // and then credit *that request's* exact sizes: the sums over the
        // position anchored at the last completion, the count naming how
        // many completions landed since. Exactly one is that request's own
        // row; more means the movement belongs to no single request the
        // panels track, and nothing claims it. (An idle scrape reads the
        // same span: on a quiet server one completion is the just-finished
        // request's — the shape the panels finish a record on.)
        let mut exact_prompt = if c.req_prompt_count - self.last_req_count > 0.5 {
            let n = c.req_prompt_count - self.last_req_count;
            if n < 1.5 {
                (c.req_prompt_sum - self.last_req_sum).max(0.0) as u64
            } else {
                0
            }
        } else {
            0
        };
        let mut exact_gen = if c.req_gen_count - self.last_gen_count > 0.5 {
            let n = c.req_gen_count - self.last_gen_count;
            if n < 1.5 {
                (c.req_gen_sum - self.last_gen_sum).max(0.0) as u64
            } else {
                0
            }
        } else {
            0
        };
        // Attribution: the per-request histograms can move while a request
        // is in flight — but only for *foreign* completions: the engine
        // credits them the moment a request completes, and the tracked one
        // has not. Anchored at the live positions of the scrape the
        // request's window opened (its admission; anything that moved up
        // to and including that scrape belongs to the request before it),
        // an unmoved histogram span proves the tracked request is the only
        // completion since — then a one-count credit is its own exact
        // sizes. Moved means a foreign completion landed mid-window: any
        // lone credit names that request, not this one (the probe's row
        // printed the churner's 2,805 that way), and on a completion-edge
        // scrape nothing here names the tracked request's spans — its
        // counter publication went unseen.
        // Attribution (two rules, both from the trace):
        // 1. Foreign credit: the histograms moved strictly after the
        //    tracked request's window opened — a completion of someone
        //    else landed mid-flight. Its exact sizes name that request.
        // 2. Window rule: the histograms move once per *completion*, so
        //    the sums can only ever name a request that has finished.
        //    While `running` is lit, whatever the histograms moved belongs
        //    to a request that completed while the tracked one was in
        //    flight — never to the tracked one (the live server printed
        //    the churner's 2,805 on the probe's row through exactly this).
        // The probe's own exact sizes ride the *idle* scrape after its
        // completion, where the one-count credit names the request that
        // just finished (rule 1 still vetoes a busy idle stretch).
        let foreign_credit = c.req_prompt_count > self.pre_window_req_count + 0.5
            || c.req_gen_count > self.pre_window_gen_count + 0.5;
        if foreign_credit {
            exact_prompt = 0;
            exact_gen = 0;
        }
        // Ledger-offset continuity: mlx-serve moves `prompt_tokens_total`
        // and the completion histogram together, so their difference is
        // constant across a request's whole life unless the counters moved
        // on a scrape between the adapter's reads. A moved offset is proof
        // the engine published and re-based in an unseen beat: nothing
        // this scrape can see brackets the finished request. (Runs before
        // any branch below re-anchors the whole-counter baselines, so the
        // same scrapes' positions feed both ledgers' comparison.)
        let completed = (c.succeeded - self.prev_succeeded).max(0.0) as i64;
        let hist_offset = c.req_prompt_sum - c.prompt_total;
        let hist_gen_offset = c.req_gen_sum - c.generation_total;
        let offset_moved = self.offset_seen
            && ((hist_offset - self.last_hist_offset).abs() > 0.5
                || (hist_gen_offset - self.last_hist_gen_offset).abs() > 0.5);
        self.last_hist_offset = hist_offset;
        self.last_hist_gen_offset = hist_gen_offset;
        self.offset_seen = true;
        if offset_moved {
            exact_prompt = 0;
            exact_gen = 0;
        } else if completed > 0 && exact_prompt > 0 {
            // The whole-counter published no foreign flood for these
            // completions and the ledgers moved together (nothing unseen):
            // a one-count credit is the completion's exact size — the
            // shape of a churner's completion scrape, whose whole-counter
            // movement the close brackets separately through the straddle
            // branch below.
        } else if completed > 0 {
            // A completion edge that read no whole-counter publication of
            // its own: the request's size went unseen, and no lone credit
            // on this scrape can stand in for it under continuous batching.
            exact_prompt = 0;
            exact_gen = 0;
        }

        if !self.primed {
            self.primed = true;
            self.prev_prompt = c.prompt_total;
            self.prev_generation = c.generation_total;
            self.prev_cached = c.cached_total;
            self.prev_succeeded = c.succeeded;
            self.base_succeeded = c.succeeded;
            self.prev_running = running;
            self.prev_ttft_sum = c.ttft_sum;
            self.prev_ttft_count = c.ttft_count;
            self.prev_itl_sum = c.itl_sum;
            self.base_prompt = c.prompt_total;
            self.base_generation = c.generation_total;
            self.base_cached = c.cached_total;
            self.base_ttft_sum = c.ttft_sum;
            self.base_ttft_count = c.ttft_count;
            self.base_itl_sum = c.itl_sum;
            self.prev_req_sum = c.req_prompt_sum;
            self.prev_gen_sum = c.req_gen_sum;
            self.prev_req_count = c.req_prompt_count;
            self.prev_gen_count = c.req_gen_count;
            self.last_req_sum = c.req_prompt_sum;
            self.last_gen_sum = c.req_gen_sum;
            self.last_req_count = c.req_prompt_count;
            self.last_gen_count = c.req_gen_count;
            self.task_seq = c.succeeded as i64 + i64::from(running);
        } else {
            // The engine restarted (or counters were reset): the baseline
            // is now ahead of the live counters. Re-anchor; perf's reset
            // clause / idle edge picks the request back up.
            if c.prompt_total < self.base_prompt - 0.5
                || c.generation_total < self.base_generation - 0.5
                || c.req_prompt_sum < self.last_req_sum - 0.5
            {
                // A server restart rewound the counters: re-anchor so
                // the next request's window does not span the restart.
                self.base_prompt = c.prompt_total;
                self.base_generation = c.generation_total;
                self.base_cached = c.cached_total;
                self.base_ttft_sum = c.ttft_sum;
                self.base_ttft_count = c.ttft_count;
                self.base_itl_sum = c.itl_sum;
                self.base_succeeded = c.succeeded;
                self.last_req_sum = c.req_prompt_sum;
                self.last_gen_sum = c.req_gen_sum;
                self.last_req_count = c.req_prompt_count;
                self.last_gen_count = c.req_gen_count;
                self.task_seq = c.succeeded as i64;
            }

            // Remember the whole-request anchors this scrape starts from:
            // if a branch below moves them, the edge scrape's closing view
            // still reads the finished request through these.
            self.edge_base_prompt = self.base_prompt;
            self.edge_base_cached = self.base_cached;
            self.edge_base_generation = self.base_generation;
            self.edge_base_ttft_sum = self.base_ttft_sum;
            self.edge_base_ttft_count = self.base_ttft_count;
            self.edge_base_itl_sum = self.base_itl_sum;

            // (`completed` and the ledger-offset state were computed at the
            // top of this scrape, before any branch moved a baseline.)
            // Mlx reveals completion twice: the engine moves its vllm-named
            // counters (prompt, generation, succeeded, TTFT) while `running`
            // stays lit under continuous batching, and `running` only goes
            // dark when nothing follows. The two edges the panels key on —
            // the counters moving and the phase going dark — can fall on
            // different scrapes, and a scrape between them is invisible:
            // whichever scrape this is, the finished request's spans must be
            // readable before any branch below re-anchors the baselines. The
            // close is staged here on the *counter* edge, where positions
            // still bracket the request, and delivered to the panels on
            // whichever scrape carries the phase going dark.
            //
            // Which scrape carried the request's counters?
            //
            // - No shortfall (`succeeded` moved by exactly what this scrape's
            //   counters moved with it, or the engine was idle since the
            //   previous scrape): the movement landed *this* scrape. vLLM's
            //   honest shape always reads this way, and so does mlx's
            //   completion when the adapter happens to poll the very scrape
            //   the engine credited.
            // - Shortfall (`succeeded` arrived ahead of any movement this
            //   scrape can account for): the counters moved on a scrape
            //   between the adapter's two reads. This scrape's positions
            //   carry whatever the engine credited since — on a cache-heavy
            //   server that flood is other requests' prefix-cache restores
            //   (the 198,720 t/s row); only the previous scrape's positions
            //   still bracket the finished request. An empty bracket means a
            //   gap nothing can reconstruct: the close stays spanless and
            //   the engine-published rates carry the row.
            //
            // The straddle shape splits once more by what actually moved.
            // When the *previous* scrape's positions read blank against the
            // admission baseline — the whole span, tokens and histogram both,
            // moved since the adapter last looked — the request finished
            // while the adapter was mid-poll, and the previous scrape was
            // the successor's first. Then this scrape's movement, over the
            // admission baseline, brackets the finished request alone (the
            // successor's counters cannot move before *its* completion), and
            // the raw reading stands. When the previous scrape carried
            // partial movement (the engine's staged publication), the raw
            // whole-counter delta mixes that request's span with the flood's
            // (the 13,302-token row the flood trace caught), and the close
            // reads through the previous scrape's positions instead. Either
            // way, a close that ends up naming no movement at all (both
            // scrapes blank, the counters moved twice unseen) is built from
            // the request's known sizes: mlx's prompt rides in from the
            // expectation `LiveRates` notes at admission, the engine's exact
            // rates from the staged pair; the spans stay blank rather than
            // crediting the row with the flood.
            if completed > 0 {
                // The request this close names: whatever the capture below
                // decides, the expectation is consumed with it. (Plain vLLM
                // never notes one, so this only matters on the mlx path.)
                let named_prompt = std::mem::take(&mut self.pending_prompt);
                // The whole-token counters' movement across the *previous*
                // scrape's positions into this one: on the completion edge
                // that delta is the finished request's exact prompt (mlx
                // moves `prompt_tokens_total` in one block when a request
                // completes — its whole size, prefix-cache restores
                // included, and the flood rides `cached_tokens_total`, a
                // different ledger). The adapter snapshots the positions at
                // the end of every scrape, exactly like perf's `prev`.
                // The whole-token counters' bracket over the *finished*
                // request: its span was published here in one block when
                // it completed. `base_*` opened at its admission and no
                // successor's completion can move this ledger before its
                // own — so the window's span is a lower bracket even when
                // the engine credited the request a scrape before the
                // phase went dark (the one-scrape delta reads blank then;
                // the window still holds the whole span).
                let published_prompt = (c.prompt_total - self.prev_prompt)
                    .max(0.0)
                    .max((c.prompt_total - self.edge_base_prompt).max(0.0))
                    as u64;
                let published_gen = (c.generation_total - self.prev_generation)
                    .max(0.0)
                    .max((c.generation_total - self.edge_base_generation).max(0.0))
                    as u64;
                let mut close = ClosingRequest {
                    // The finished request's ordinal: `succeeded` counts
                    // completions, and the close names the request that
                    // just completed. The panels' open record still wears
                    // this id even when the adapter has re-numbered its
                    // task sequence for a successor that lit the same
                    // scrape — the identity check that keeps a straddled
                    // close off someone else's row.
                    // The successor's task number (this close is built on
                    // the scrape that re-numbers for the request that just
                    // lit): the panels' open record carries the *finished*
                    // request's id, so perf's whole-span substitutions
                    // refuse this close and read through its exact pair
                    // instead. (The close's spans come from the whole-
                    // counter window, which a busy server inflates; its
                    // exact pair and measured rates are per-request.)
                    id_task: self.task_seq.max(1),
                    prompt: (c.prompt_total - self.edge_base_prompt).max(0.0) as usize,
                    cached: (c.cached_total - self.edge_base_cached).max(0.0) as usize,
                    gen: (c.generation_total - self.edge_base_generation).max(0.0) as usize,
                    prefill_secs: self.pending_prefill_secs.take(),
                    measured_prefill_tps: self.pending_measured_rate.take(),
                    measured_decode_tps: self.pending_measured_decode.take(),
                    // Mean TTFT over the completions in this window
                    // (exact when exactly one completed).
                    ttft_secs: (c.ttft_sum - self.edge_base_ttft_sum).max(0.0)
                        / (c.ttft_count - self.edge_base_ttft_count).max(1.0),
                    itl_sum: (c.itl_sum - self.edge_base_itl_sum).max(0.0),
                    // The request's exact token counts when exactly one
                    // completion landed since the last completion the
                    // adapter anchored: `exact_prompt` above reads the
                    // histograms over exactly that span.
                    exact_prompt,
                    exact_gen,
                    published_prompt,
                    published_gen,
                    // The whole-token ledger moved across exactly this
                    // scrape's completion edge.
                    published_completions: completed.max(0) as u64,
                    // The engine's live gauges named this request's prompt
                    // whole while it ran (the `note_measured_prompt`
                    // ledger); a close whose spans outrun it absorbed
                    // another request's publication. 0 for engines that
                    // publish no live expectation (vLLM, strata).
                    span_known: self.pending_span_size,
                };
                // Task identity: the close names the request that completed
                // (`succeeded` counts completions, so the request finished
                // this scrape is number `succeeded`). The panels' open
                // record is keyed by `task_seq` — which is the same number
                // while the phase stays lit (continuous batching numbers
                // requests by admission, not completion, and the adapter
                // re-numbers for a successor only through this close). Keep
                // the two ledgers honest: perf's rescue keys its whole-span
                // substitutions on this id agreeing with the record's own.
                close.id_task = self.task_seq.max(1) - 1;
                // The published span belonged to the request that just
                // closed; this scrape's expectation (lit on the same
                // scrape under continuous batching) opens the successor's
                // window, so the span ledger starts blank for it.
                self.pending_span_size = 0;
                let prev_span = (self.prev_prompt - self.edge_base_prompt).max(0.0);
                let prev_gen = (self.prev_generation - self.edge_base_generation).max(0.0);
                if completed == 1 && prev_span > 0.0 {
                    // The straddle: the request's own span was published to
                    // the previous scrape (churner shape: the engine's staged
                    // publication lands one poll before the edge sees the
                    // completion). That scrape's positions still bracket the
                    // finished request — the successor was admitted this
                    // scrape and has moved nothing — while this scrape's
                    // carry the flood on top of them.
                    close.prompt = prev_span as usize;
                    close.cached = (self.prev_cached - self.edge_base_cached).max(0.0) as usize;
                    close.gen = prev_gen as usize;
                    // The engine's measured pair survives the straddle in
                    // the pendings (mlx's LiveRates stages them on the
                    // completion scrape, whatever the phase reads): a close
                    // whose spans came from the previous scrape's positions
                    // still hands the panels this request's exact rate and
                    // span, not the raw reading's (which is the successor's
                    // or blank).
                    close.prefill_secs = self.pending_prefill_secs;
                    close.measured_prefill_tps = self.pending_measured_rate;
                    close.measured_decode_tps = self.pending_measured_decode;
                    close.ttft_secs = (self.prev_ttft_sum - self.edge_base_ttft_sum).max(0.0)
                        / (self.prev_ttft_count - self.edge_base_ttft_count).max(1.0);
                    close.itl_sum = (self.prev_itl_sum - self.edge_base_itl_sum).max(0.0);
                    // The exact pair belongs with the *other* bracket: a
                    // previous scrape that moved the histograms by exactly
                    // one completion names the request whose span landed
                    // there — the request the engine published, not the
                    // successor this scrape's counters carry. (Reading the
                    // raw pair here is what mislabelled the probe's row
                    // with the churner's 2,805.)
                }
                // The exact pair is only honest for a close that carries
                // the whole window the sums span: no foreign completion
                // moved the histograms while the request was in flight.
                // A foreign credit names some other request — the churn
                // shape's probe printed the churner's 2,805 through it.
                if foreign_credit || (close.prompt == 0 && close.gen == 0) {
                    // No span of its own: the exact pair cannot name this
                    // request (a lone credit over any window is some other
                    // completion's under churn).
                    close.exact_prompt = 0;
                    close.exact_gen = 0;
                }
                if named_prompt > 0 && self.pending_span_known && close.prompt > named_prompt {
                    // The capture's whole-counter span outruns the prompt
                    // the engine itself published for the request: foreign
                    // movement inflated it. The published span is exact —
                    // the live gauges carry the prompt whole at admission —
                    // so the row reads through it, and the rescue's rates
                    // divide honest numerators.
                    close.prompt = named_prompt;
                    close.cached = close.cached.min(close.prompt);
                    if close.gen == 0 {
                        close.gen = std::mem::take(&mut self.pending_gen);
                    }
                }
                if close.prompt == 0 && close.gen == 0 && !foreign_credit {
                    // A close that names no movement at all (the counters
                    // moved between polls unseen) still names a request
                    // when mlx noted its prompt from the live gauges while
                    // *it* ran. Take that memory as the span rather than
                    // print the flood's sum or nothing at all — but only
                    // the in-flight variety: under continuous batching a
                    // close can build while the *successor* is lit, and a
                    // completion's residual sizes would name the wrong
                    // request entirely.
                    if named_prompt > 0 && self.pending_in_flight {
                        close.prompt = named_prompt;
                        close.cached = close.cached.min(close.prompt);
                        if close.gen == 0 {
                            close.gen = std::mem::take(&mut self.pending_gen);
                        }
                    }
                }
                self.pending_in_flight = false;
                self.pending_span_known = false;
                // The per-request histograms moved with these completions:
                // advance the exact-position anchors so the next request
                // reads its own sum from here. (Mlx's engine credits the
                // histograms while `running` stays lit, so the adapter's
                // phase edge alone cannot be the only place they move.)
                self.last_req_sum = c.req_prompt_sum;
                self.last_gen_sum = c.req_gen_sum;
                self.last_req_count = c.req_prompt_count;
                self.last_gen_count = c.req_gen_count;
                if running {
                    // A successor is running: re-anchor to this sample —
                    // perf.rs re-reads counters wholesale on a new task.
                    // The finished request's spans are captured above.
                    self.base_prompt = c.prompt_total;
                    self.base_generation = c.generation_total;
                    self.base_cached = c.cached_total;
                    self.base_ttft_sum = c.ttft_sum;
                    self.base_ttft_count = c.ttft_count;
                    self.base_itl_sum = c.itl_sum;
                    self.base_succeeded = c.succeeded;
                    // The successor's exact anchors open at this scrape's
                    // positions: everything that moved before them belongs
                    // to the finished request (whose close is built above),
                    // and its own completion cannot move the histograms
                    // before *its* completion scrape. Foreign credits
                    // during its life show as movement over these anchors
                    // and are refused, not misread as the successor's size.
                    self.last_req_sum = c.req_prompt_sum;
                    self.last_gen_sum = c.req_gen_sum;
                    self.last_req_count = c.req_prompt_count;
                    self.last_gen_count = c.req_gen_count;
                    self.pre_window_req_sum = c.req_prompt_sum; // same scrape
                    self.pre_window_gen_sum = c.req_gen_sum;
                    self.pre_window_req_count = c.req_prompt_count;
                    self.pre_window_gen_count = c.req_gen_count;
                    self.task_seq = c.succeeded as i64 + 1;
                } else {
                    // The last request finished and nothing followed: keep
                    // the old baseline so the closing sample still carries
                    // the full request.
                    self.task_seq = c.succeeded as i64;
                }
                if close.prompt > 0
                    || close.gen > 0
                    || close.measured_prefill_tps.is_some()
                    || close.measured_decode_tps.is_some()
                {
                    self.staged_close = Some(close);
                }
            } else if running && !self.prev_running {
                // A fresh admission straight from idle: the whole poll
                // window belongs to the new request.
                self.base_prompt = self.prev_prompt;
                self.base_generation = self.prev_generation;
                self.base_cached = self.prev_cached;
                self.base_ttft_sum = self.prev_ttft_sum;
                self.base_ttft_count = self.prev_ttft_count;
                self.base_itl_sum = self.prev_itl_sum;
                self.base_succeeded = self.prev_succeeded;
                // The new request's exact anchors open at the *previous*
                // scrape's positions — the last that predate its admission,
                // exactly like the whole-request baselines above. The
                // engine credits a request's sizes at its completion, which
                // the adapter sees one scrape late (under churn the success
                // edge and the counter publication straddle), so a
                // completion-edge scrape must still bracket: `req_count >
                // pre_window_count` names a foreign completion that landed
                // strictly after the request's admission, while the
                // request's own credit lands on the bracket's far edge.
                self.last_req_sum = self.prev_req_sum;
                self.last_gen_sum = self.prev_gen_sum;
                self.last_req_count = self.prev_req_count;
                self.last_gen_count = self.prev_gen_count;
                self.pre_window_req_sum = self.prev_req_sum;
                self.pre_window_gen_sum = self.prev_gen_sum;
                self.pre_window_req_count = self.prev_req_count;
                self.pre_window_gen_count = self.prev_gen_count;
                self.task_seq = c.succeeded as i64 + 1;
            }

            // Mlx reveals completion twice: the engine moves its vllm-named
            // counters (prompt, generation, succeeded, TTFT) while `running`
            // stays lit under continuous batching, and `running` only goes
            // dark when nothing follows. The close above is staged on the
            // *counter* edge, where the positions still bracket the finished
            // request; deliver it to the panels on whichever scrape carries
            // the phase going dark. A scrape that moves the counters without
            // ending the phase stages the close and hands nothing yet.
            if self.pending_close.is_none() {
                let close = self.staged_close.take().or_else(|| {
                    // The idle edge without a counter edge: vLLM moves its
                    // counters exactly when `running` goes dark, so this
                    // scrape's positions over the admitted-request baseline
                    // *are* the request's whole span — and the counters of
                    // any successor still in the queue have not moved.
                    // Mlx's `running` goes dark for reasons of its own; its
                    // completions always move `succeeded` on an earlier
                    // scrape, so an idle edge there that carried no staged
                    // close names no request and builds blank.
                    (completed == 0 && !running && self.prev_running)
                        .then(|| {
                            let close = ClosingRequest {
                                id_task: c.succeeded.max(1.0) as i64 - 1,
                                prompt: (c.prompt_total - self.base_prompt).max(0.0) as usize,
                                cached: (c.cached_total - self.base_cached).max(0.0) as usize,
                                gen: (c.generation_total - self.base_generation).max(0.0) as usize,
                                prefill_secs: self.pending_prefill_secs.take(),
                                measured_prefill_tps: self.pending_measured_rate.take(),
                                measured_decode_tps: self.pending_measured_decode.take(),
                                ttft_secs: (c.ttft_sum - self.base_ttft_sum).max(0.0)
                                    / (c.ttft_count - self.base_ttft_count).max(1.0),
                                itl_sum: (c.itl_sum - self.base_itl_sum).max(0.0),
                                exact_prompt,
                                exact_gen,
                                published_prompt: (c.prompt_total - self.prev_prompt)
                                    .max(0.0)
                                    .max((c.prompt_total - self.base_prompt).max(0.0))
                                    as u64,
                                published_gen: (c.generation_total - self.prev_generation)
                                    .max(0.0)
                                    .max((c.generation_total - self.base_generation).max(0.0))
                                    as u64,
                                // Completions across the window this close's
                                // spans span (the admitted request's whole
                                // life): mlx publishes one request's whole
                                // span per completion, so a count of exactly
                                // one makes the spans this request's alone.
                                published_completions: (c.succeeded - self.base_succeeded).max(0.0)
                                    as u64,
                                // Idle-edge closes name no request under
                                // continuous batching (mlx's completions
                                // always ride a counter edge first), so no
                                // published span either.
                                span_known: 0,
                            };
                            if close.prompt == 0
                                && close.gen == 0
                                && close.measured_prefill_tps.is_none()
                                && close.measured_decode_tps.is_none()
                            {
                                // Nothing to report: keep the engine's rates
                                // alive for the next edge instead of eating them.
                                self.pending_prefill_secs = close.prefill_secs;
                                self.pending_measured_rate = close.measured_prefill_tps;
                                self.pending_measured_decode = close.measured_decode_tps;
                                None
                            } else {
                                Some(close)
                            }
                        })
                        .flatten()
                });
                if let Some(close) = close {
                    self.pending_close = Some(close);
                }
            }

            self.prev_prompt = c.prompt_total;
            self.prev_generation = c.generation_total;
            self.prev_cached = c.cached_total;
            self.prev_succeeded = c.succeeded;
            self.prev_running = running;
            self.prev_ttft_sum = c.ttft_sum;
            self.prev_ttft_count = c.ttft_count;
            self.prev_itl_sum = c.itl_sum;
            self.prev_req_sum = c.req_prompt_sum;
            self.prev_gen_sum = c.req_gen_sum;
            self.prev_req_count = c.req_prompt_count;
            self.prev_gen_count = c.req_gen_count;
        }

        let prompt_req = (c.prompt_total - self.base_prompt).max(0.0) as usize;
        let cached_req = (c.cached_total - self.base_cached).max(0.0) as usize;
        let decoded_req = (c.generation_total - self.base_generation).max(0.0) as usize;
        let spec_active = c.spec_drafts > 0.0 || c.spec_accepted > 0.0;

        let s = LiveStats {
            ctx_max: 0,
            prompt_tokens: prompt_req + cached_req,
            prompt_processed: prompt_req,
            prefill_tps: None,
            prefill_rate_live: false,
            // The engine's measured pair lives in LiveRates (it comes from
            // mlx's live gauges, not vllm's counters), which stamps these
            // three onto the stats *after* this literal, on the scrapes its
            // window model publishes them. Plain vLLM's servers publish no
            // such gauges and they stay None.
            measured_prefill_tps: None,
            measured_decode_tps: None,
            prefill_secs: None,
            decoded: decoded_req,
            decoded_present: true,
            cache_tokens: cached_req,
            processing: running,
            spec_types: if spec_active {
                c.spec_backend.clone().unwrap_or_else(|| "mtp".to_string())
            } else {
                "none".to_string()
            },
            id_task: self.task_seq,
            n_slots: 0,
            slots_busy: 0,
            spec_depth: c.spec_positions.len(),
            // vLLM moves its token/latency counters only at completion:
            // while a request is in flight these are 0, and the poll
            // that closed it carries the request's whole measured
            // lifetime (mean TTFT over the window's completions, and
            // the sum of per-request ITLs).
            ttft_secs: (c.ttft_sum - self.base_ttft_sum).max(0.0)
                / (c.ttft_count - self.base_ttft_count).max(1.0),
            itl_sum: (c.itl_sum - self.base_itl_sum).max(0.0),
            exact_prompt,
            exact_gen,
            // The whole-token counters' movement since the previous scrape,
            // attributed: mlx moves that ledger one completion's sizes at a
            // time, so only a scrape that moves the completion count too
            // names a request — and only a count delta of exactly one names
            // a *single* request the panels can point at (a batch credits
            // no row alone). A lit scrape's pair describes a request that
            // *completed* while another is in flight; perf bills it to the
            // finished row through the closing view, never to the live one.
            published_prompt: if completed == 1 {
                (c.prompt_total - self.prev_prompt).max(0.0) as u64
            } else {
                0
            },
            published_gen: if completed == 1 {
                (c.generation_total - self.prev_generation).max(0.0) as u64
            } else {
                0
            },
            closing: self.pending_close.take(),
            cache_unknown: false,
            weight_gb: None,
            kv_cache_gb: None,
            graph_gb: None,
            kv_tokens: None,
        };

        let spec = spec_active.then_some(SpecMetrics {
            draft_tokens: c.spec_draft_tokens as u64,
            accepted: c.spec_accepted as u64,
            verify_steps: c.spec_drafts as u64,
            n_decode: c.generation_total as u64,
            tokens_predicted: c.generation_total as u64,
            busy_secs: 0.0,
        });

        (s, spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(prompt: f64, gen: f64, running: f64, succeeded: f64) -> VllmCounters {
        VllmCounters {
            prompt_total: prompt,
            generation_total: gen,
            cached_total: 0.0,
            running,
            succeeded,
            ..Default::default()
        }
    }

    #[test]
    fn baseline_and_request_flips() {
        let mut a = VllmAdapter::new();

        // Idle, engine already ran traffic before the adapter attached.
        let (s, spec) = a.observe(&counters(100.0, 50.0, 0.0, 7.0));
        assert!(!s.processing);
        assert_eq!(s.id_task, 7);
        assert_eq!(s.decoded, 0);
        assert!(spec.is_none());

        // A request is admitted on the next poll: the window's tokens go to it.
        let (s2, _) = a.observe(&counters(160.0, 50.0, 1.0, 7.0));
        assert!(s2.processing);
        assert_eq!(s2.id_task, 8);
        assert_eq!(s2.prompt_processed, 60);
        assert_eq!(s2.decoded, 0);

        // It completes in the same poll window and a successor is admitted.
        let (s3, _) = a.observe(&counters(300.0, 120.0, 1.0, 8.0));
        assert!(s3.processing);
        assert_eq!(s3.id_task, 9);
        assert_eq!(s3.decoded, 0); // re-anchored to the current sample

        // And it goes idle again.
        let (s4, _) = a.observe(&counters(300.0, 120.0, 0.0, 9.0));
        assert!(!s4.processing);
        assert_eq!(s4.id_task, 9);
        assert_eq!(s4.decoded, 0);
        assert_eq!(s4.prompt_tokens, 0);
        for stats in [&s, &s2, &s3, &s4] {
            assert!(stats.prefill_tps.is_none());
            assert!(stats.prefill_secs.is_none());
            if let Some(closing) = &stats.closing {
                assert!(closing.prefill_secs.is_none());
            }
        }
    }

    #[test]
    fn prefill_and_decode_keep_vllm_histogram_measurements() {
        use crate::perf::PerfTracker;
        use std::time::{Duration, Instant};

        let mut adapter = VllmAdapter::new();
        let mut perf = PerfTracker::new();
        let now = Instant::now();
        let counters = [
            counters(100.0, 50.0, 0.0, 7.0),
            counters(100.0, 50.0, 1.0, 7.0),
            VllmCounters {
                prompt_total: 700.0,
                generation_total: 90.0,
                cached_total: 200.0,
                succeeded: 8.0,
                ttft_sum: 2.0,
                ttft_count: 1.0,
                itl_sum: 1.0,
                ..Default::default()
            },
        ];
        for (i, counters) in counters.iter().enumerate() {
            let (stats, _) = adapter.observe(counters);
            assert!(stats.prefill_tps.is_none());
            assert!(stats.prefill_secs.is_none());
            perf.observe(&stats, now + Duration::from_millis(400 * i as u64));
        }
        let request = perf.history.back().unwrap();
        assert_eq!(request.prefill_tokens, 600);
        assert_eq!(request.cached_tokens, 200);
        assert_eq!(request.decoded, 40);
        assert_eq!(request.avg_prefill_tps(), 300.0);
        assert_eq!(request.avg_decode_tps(), 39.0);
        assert!(request.measured_prefill_tps.is_none());
        assert_eq!(perf.peak_prefill_tps, 300.0);
        assert_eq!(perf.peak_decode_tps, 39.0);
        assert_eq!(perf.session_prefilled, 600);
        assert_eq!(perf.session_decoded, 40);
    }

    #[test]
    fn restart_reanchors_baselines() {
        let mut a = VllmAdapter::new();
        a.observe(&counters(5000.0, 9000.0, 0.0, 40.0));
        // Engine restarts: counters rewind below the baseline.
        let (s, _) = a.observe(&counters(10.0, 0.0, 0.0, 0.0));
        assert!(!s.processing);
        assert_eq!(s.id_task, 0);
        assert_eq!(s.decoded, 0);
        assert_eq!(s.prompt_tokens, 0);
    }

    #[test]
    fn parses_real_brain_fixture() {
        // A /metrics capture from a vLLM server on an Intel Arc B70
        // (vLLM XPU), trimmed to the metric families this adapter parses.
        // The docker test harness mounts the full capture at /lv; the
        // trimmed copy committed under fixtures/ keeps the test green in a
        // plain checkout.
        let body = std::fs::read_to_string("/lv/metrics8000_base.txt")
            .or_else(|_| std::fs::read_to_string("fixtures/vllm-brain-metrics.txt"))
            .ok();
        let Some(body) = body else {
            eprintln!("skipped: no metrics fixture available");
            return;
        };
        let c = parse_vllm_metrics(&body).expect("vllm: counters present");
        assert_eq!(c.model_name.as_deref(), Some("qwen38"));
        assert!(c.prompt_total > 0.0 && c.generation_total > 0.0);
        assert_eq!(c.spec_positions.len(), 4, "MTP depth 4 expected");
        // The latency histograms the prefill/decode rates are computed
        // from (they do not move until completion).
        assert!(c.ttft_count > 0.0, "fixture carries the TTFT histogram");
        assert!(c.itl_sum > 0.0, "fixture carries the ITL histogram");
    }

    #[test]
    fn guard_rejects_cross_wired_port() {
        // Body as served by a model called qwen38 on a port that another
        // detected model (expected "model") also claims.
        let body = "# HELP x\nvllm:prompt_tokens_total{engine=\"0\",model_name=\"qwen38\"} 10.0\nvllm:generation_tokens_total{engine=\"0\",model_name=\"qwen38\"} 5.0\n";
        let c = parse_vllm_metrics(body).unwrap();
        assert_eq!(c.model_name.as_deref(), Some("qwen38"));
        let others = vec![("qwen38".to_string(), 8000u16)];
        let label = c.model_name.as_deref().unwrap();
        assert!(cross_wired(label, "model", 8000, &others));
        // Same port, unrelated names: no cross-wiring evidence, accept.
        assert!(!cross_wired(label, "model", 8001, &others));
        // Matching name: accept.
        assert!(!cross_wired(label, "qwen38", 8000, &others));
    }
}
