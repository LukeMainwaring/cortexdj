import type { NextRequest } from "next/server";

import { BACKEND_URL } from "@/lib/constants";

export async function POST(request: NextRequest) {
  const body = await request.text();

  const response = await fetch(`${BACKEND_URL}/api/agent/chat`, {
    body,
    headers: {
      "Content-Type": "application/json",
    },
    method: "POST",
  });

  // A failure passes through with the backend's status and body, so the JSON
  // `detail` FastAPI returns reaches the UI unchanged.
  return new Response(response.body, {
    headers: {
      "Content-Type": response.headers.get("Content-Type") || "text/plain",
    },
    status: response.status,
  });
}
