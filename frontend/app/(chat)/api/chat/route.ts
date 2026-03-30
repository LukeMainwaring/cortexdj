import { type NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8003";

export async function POST(request: NextRequest) {
  const body = await request.text();

  const response = await fetch(`${BACKEND_URL}/api/agent/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body,
  });

  if (!response.ok) {
    return NextResponse.json(
      { error: "Backend request failed" },
      { status: response.status },
    );
  }

  return new Response(response.body, {
    headers: {
      "Content-Type": response.headers.get("Content-Type") || "text/plain",
    },
  });
}
