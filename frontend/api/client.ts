import { client } from "./generated/client.gen";

client.setConfig({
  baseURL: process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8003",
  withCredentials: true,
});
