export function extractSessionId(input: unknown): string | null {
  if (input && typeof input === "object" && "session_id" in input) {
    const value = (input as { session_id: unknown }).session_id;
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return null;
}
