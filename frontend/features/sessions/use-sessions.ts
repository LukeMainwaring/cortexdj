import { useQuery } from "@tanstack/react-query";
import {
  getBrainContextOptions,
  getSessionSegmentsOptions,
  listSessionsEnrichedOptions,
} from "@/api/generated/@tanstack/react-query.gen";

// Fetch a generous slice so any subset the agent might choose is covered by
// the cached query. 500 is well above the demo's ceiling (32 DEAP rows) and
// still a single query — if the catalog ever scales past this we should
// fetch the requested ids directly instead of filtering client-side.
const ENRICHED_SESSIONS_LIMIT = 500;

export const useEnrichedSessions = () =>
  useQuery({
    ...listSessionsEnrichedOptions({
      query: { limit: ENRICHED_SESSIONS_LIMIT },
    }),
    staleTime: 60_000,
  });

export const useSessionSegments = (sessionId: string) =>
  useQuery(getSessionSegmentsOptions({ path: { session_id: sessionId } }));

// The brain context the agent pinned to this thread with `set_brain_context`,
// or null. A thread that doesn't exist yet also answers null, so the query
// needs no gate on a new chat. A finished turn refreshes it through
// useInvalidateThread, which invalidates every query keyed by the thread id.
export const useBrainContext = (threadId: string) =>
  useQuery(getBrainContextOptions({ path: { thread_id: threadId } }));
