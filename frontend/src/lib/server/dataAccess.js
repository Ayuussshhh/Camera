import { getDatabaseConnection } from "@/lib/server/databaseConfig";

export function getDatabaseStatus() {
  const databaseConnection = getDatabaseConnection();

  return {
    mode: databaseConnection.configured ? "postgres" : "unconfigured",
    postgresEnabled: databaseConnection.configured,
    source: databaseConnection.mode,
    host: databaseConnection.host || null,
    port: databaseConnection.port || null,
    database: databaseConnection.database || null
  };
}

export function getAiServiceUrl() {
  return (
    process.env.AI_SERVICE_URL ||
    process.env.NEXT_PUBLIC_AI_SERVICE_URL ||
    "http://127.0.0.1:8000"
  );
}

export async function safeFetchJson(url, options = {}, timeoutMs = 4000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {})
      }
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(payload.message || `Request failed with ${response.status}`);
    }

    return payload;
  } finally {
    clearTimeout(timeout);
  }
}
