import { BACKEND_URL } from "@/lib/constants";
import { client } from "./generated/client.gen";

client.setConfig({
  baseURL: BACKEND_URL,
  withCredentials: true,
});
