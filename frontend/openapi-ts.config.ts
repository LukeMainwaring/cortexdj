import { defineConfig } from "@hey-api/openapi-ts";

export default defineConfig({
  input: {
    path: "api/scripts/outputs/openapi.json",
  },
  output: {
    path: "api/generated",
  },
  parser: {
    filters: {
      // Streaming chat uses the AI SDK transport, not the REST client.
      operations: { exclude: ["POST /api/agent/chat"] },
      tags: { exclude: ["health"] },
    },
  },
  plugins: [
    "@tanstack/react-query",
    {
      name: "@hey-api/client-axios",
      runtimeConfigPath: "@/api/client",
    },
  ],
});
