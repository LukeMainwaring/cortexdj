import { BACKEND_URL } from "@/lib/constants";
import type { CreateClientConfig } from "./generated/client.gen";

/** Initialize every SDK consumer with the same backend URL and credentials. */
export const createClientConfig: CreateClientConfig = (config) => ({
  ...config,
  baseURL: BACKEND_URL,
  withCredentials: true,
});
