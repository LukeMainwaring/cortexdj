import fs from "fs";

const DEFAULT_API_URL = "http://localhost:8003/api/openapi.json";
const OUTPUT_PATH = "./api/scripts/outputs/openapi.json";

async function fetchOpenApiSpec(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch OpenAPI spec: ${response.status}`);
  }
  return response.json();
}

async function main() {
  const apiUrl = process.argv[2] || DEFAULT_API_URL;

  try {
    const spec = await fetchOpenApiSpec(apiUrl);

    await fs.promises.writeFile(
      OUTPUT_PATH,
      JSON.stringify(spec, null, 2) + "\n"
    );
    console.info(`OpenAPI spec saved to ${OUTPUT_PATH}`);
  } catch (err) {
    console.error("Error:", err.message);
    process.exit(1);
  }
}

main();
