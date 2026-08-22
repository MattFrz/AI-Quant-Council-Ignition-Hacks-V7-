/**
 * frontend/lib/stream.ts — D2
 *
 * EventSource consumer for ResearchEvent (the D6 live timeline). In fixture
 * mode it replays the 13-step pipeline from PIPELINE_STEPS on a timer instead
 * of opening a real connection - the UI can't tell the difference, and
 * neither can a rehearsal. In live mode it opens a real EventSource against
 * the SSE endpoint from Phase 3 step 3.4.
 *
 * Both modes are driven off the same USE_FIXTURE flag as api.ts, so D1 and D2
 * flip together in Phase 3 step 3.8.
 */
import { PIPELINE_STEPS, STEP_LABELS } from "./types";
import type { ResearchEvent, StepStatus } from "./types";
import { API_BASE, USE_FIXTURE, FIXTURE_TRADE_IDEA } from "./api";

export type ResearchEventHandler = (event: ResearchEvent) => void;

export interface ResearchStreamHandle {
  close: () => void;
}

// Sentinel stream_url api.ts hands back in fixture mode (see submitThesis).
export const FIXTURE_STREAM_URL = "fixture://replay";

// Optional per-step detail text for the fixture replay, so the timeline reads
// like a real run instead of thirteen bare labels. Anything left out falls
// back to just the label with no detail.
const FIXTURE_STEP_DETAIL: Partial<Record<string, string>> = {
  scan_universe: "Scanned 1,247 companies against liquidity and market-cap filters",
  identify_candidates: "Narrowed to 7 candidates after fundamental and event screens",
  retrieve_filings: `Retrieved ${FIXTURE_TRADE_IDEA.catalysts.length} source filings`,
  extract_catalysts: `Extracted ${FIXTURE_TRADE_IDEA.catalysts.length} catalysts, all source-linked`,
  backtest_signal: `Walk-forward Sharpe ${FIXTURE_TRADE_IDEA.backtest?.sharpe ?? "-"}`,
  calculate_risk: `Risk band: ${FIXTURE_TRADE_IDEA.risk?.risk_band ?? "-"}`,
  final_recommendation: `${FIXTURE_TRADE_IDEA.ticker} ${FIXTURE_TRADE_IDEA.side}, alpha ${FIXTURE_TRADE_IDEA.alpha_score}/10`,
};

// Pace of the fake replay. Each step goes pending -> running -> done, so a
// full 13-step run takes roughly PIPELINE_STEPS.length * STEP_MS. Tune this
// for demo feel - slower reads as more "real," faster keeps rehearsals quick.
const STEP_MS = 650;

/**
 * Subscribe to the research timeline for a job. Pass the stream_url straight
 * off the JobHandle api.ts gave you - this function decides fixture vs. live
 * on its own, callers don't need to know which mode is active.
 */
export function subscribeToResearchEvents(
  streamUrl: string,
  onEvent: ResearchEventHandler,
  onDone?: () => void,
  onError?: (err: Event | Error) => void,
): ResearchStreamHandle {
  if (USE_FIXTURE || streamUrl === FIXTURE_STREAM_URL) {
    return replayFixtureTimeline(onEvent, onDone);
  }
  return openLiveStream(streamUrl, onEvent, onDone, onError);
}

function makeEvent(stepId: string, status: StepStatus, detail: string | null): ResearchEvent {
  return {
    step_id: stepId,
    label: STEP_LABELS[stepId] ?? stepId,
    status,
    detail,
    progress: null,
    timestamp: new Date().toISOString(),
  };
}

function replayFixtureTimeline(
  onEvent: ResearchEventHandler,
  onDone?: () => void,
): ResearchStreamHandle {
  let cancelled = false;
  let stepIndex = 0;
  let timeoutId: ReturnType<typeof setTimeout>;

  const tick = () => {
    if (cancelled) return;

    if (stepIndex >= PIPELINE_STEPS.length) {
      onDone?.();
      return;
    }

    const stepId = PIPELINE_STEPS[stepIndex];

    // Half a step: show it running (the row lights up but isn't checked yet).
    onEvent(makeEvent(stepId, "running", null));

    timeoutId = setTimeout(() => {
      if (cancelled) return;
      // Second half: mark it done with whatever detail text we have for it.
      onEvent(makeEvent(stepId, "done", FIXTURE_STEP_DETAIL[stepId] ?? null));
      stepIndex += 1;
      timeoutId = setTimeout(tick, STEP_MS / 2);
    }, STEP_MS / 2);
  };

  timeoutId = setTimeout(tick, STEP_MS / 2);

  return {
    close: () => {
      cancelled = true;
      clearTimeout(timeoutId);
    },
  };
}

function openLiveStream(
  streamUrl: string,
  onEvent: ResearchEventHandler,
  onDone?: () => void,
  onError?: (err: Event | Error) => void,
): ResearchStreamHandle {
  const url = streamUrl.startsWith("http") ? streamUrl : `${API_BASE}${streamUrl}`;
  const source = new EventSource(url);

  source.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data) as ResearchEvent;
      onEvent(event);
      if (event.step_id === "final_recommendation" && event.status === "done") {
        onDone?.();
        source.close();
      }
    } catch (err) {
      onError?.(err as Error);
    }
  };

  source.onerror = (err) => {
    onError?.(err);
    // Browsers auto-retry EventSource on error; if the backend is actually
    // down that just spams reconnects, so shut it down instead.
    source.close();
  };

  return {
    close: () => source.close(),
  };
}
