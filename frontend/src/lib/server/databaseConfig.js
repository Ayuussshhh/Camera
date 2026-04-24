function readEnvValue(name) {
  const value = process.env[name];

  if (typeof value !== "string") {
    return "";
  }

  return value.trim();
}

function buildDatabaseUrl({ host, port, database, user, password }) {
  const normalizedPort = port || "5432";

  return `postgresql://${encodeURIComponent(user)}:${encodeURIComponent(
    password
  )}@${host}:${normalizedPort}/${encodeURIComponent(database)}`;
}

export function getDatabaseConnection() {
  const host = readEnvValue("DB_HOST");
  const port = readEnvValue("DB_PORT") || "5432";
  const database = readEnvValue("DB_NAME");
  const user = readEnvValue("DB_USER");
  const password = readEnvValue("DB_PASSWORD");

  if (host && database && user && password) {
    return {
      configured: true,
      mode: "split_env",
      host,
      port,
      database,
      user,
      connectionString: buildDatabaseUrl({
        host,
        port,
        database,
        user,
        password
      })
    };
  }

  const databaseUrl = readEnvValue("DATABASE_URL");

  if (databaseUrl) {
    return {
      configured: true,
      mode: "database_url",
      connectionString: databaseUrl
    };
  }

  return {
    configured: false,
    mode: "unconfigured",
    connectionString: ""
  };
}
