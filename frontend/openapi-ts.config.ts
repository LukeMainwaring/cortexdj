import { defineConfig } from "@hey-api/openapi-ts";

export default defineConfig({
  input: {
    path: "api/scripts/outputs/openapi.json",
    exclude: "^/api/health",
  },
  output: {
    path: "api/generated",
  },
  plugins: ["@tanstack/react-query", "@hey-api/client-axios"],
});
