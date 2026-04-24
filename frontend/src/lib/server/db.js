import { randomUUID } from "node:crypto";
import { Pool } from "pg";
import { getDatabaseConnection } from "@/lib/server/databaseConfig";
import { CANONICAL_ENGINE, normalizeEngineName } from "@/utils/engine";

const DB_POOL_KEY = "__faceTraceDbPool";
const DB_INIT_KEY = "__faceTraceDbInit";
const PLATFORM_SETTINGS_ID = "platform";

function assertDatabaseConfigured() {
  if (!getDatabaseConnection().configured) {
    throw new Error(
      "PostgreSQL is not configured. Set DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD in frontend/.env."
    );
  }
}

function getPool() {
  assertDatabaseConfigured();
  const databaseConnection = getDatabaseConnection();

  if (!globalThis[DB_POOL_KEY]) {
    globalThis[DB_POOL_KEY] = new Pool({
      connectionString: databaseConnection.connectionString
    });
  }

  return globalThis[DB_POOL_KEY];
}

async function initializeDatabase() {
  const pool = getPool();
  const client = await pool.connect();

  try {
    await client.query(`
      CREATE TABLE IF NOT EXISTS platform_settings (
        id TEXT PRIMARY KEY,
        active_engine VARCHAR(40) NOT NULL DEFAULT 'mediapipe',
        confidence_threshold NUMERIC(5, 4) NOT NULL DEFAULT 0.91,
        duplicate_window_minutes INTEGER NOT NULL DEFAULT 180,
        liveness_enabled BOOLEAN NOT NULL DEFAULT FALSE,
        anti_spoof_mode VARCHAR(32) NOT NULL DEFAULT 'ready',
        auto_scan_interval_ms INTEGER NOT NULL DEFAULT 4000,
        default_camera VARCHAR(120) NOT NULL DEFAULT 'Integrated HD Camera',
        camera_enabled BOOLEAN NOT NULL DEFAULT TRUE,
        notifications_enabled BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await client.query(`
      ALTER TABLE platform_settings
      ALTER COLUMN active_engine SET DEFAULT 'mediapipe'
    `);

    await client.query(`
      ALTER TABLE platform_settings
      ALTER COLUMN confidence_threshold SET DEFAULT 0.91
    `);

    await client.query(
      `
        INSERT INTO platform_settings (
          id,
          active_engine,
          confidence_threshold,
          duplicate_window_minutes,
          liveness_enabled,
          anti_spoof_mode,
          auto_scan_interval_ms,
          default_camera,
          camera_enabled,
          notifications_enabled
        )
        VALUES (
          $1,
          $2,
          0.91,
          180,
          FALSE,
          'ready',
          4000,
          'Integrated HD Camera',
          TRUE,
          TRUE
        )
        ON CONFLICT (id) DO NOTHING
      `,
      [PLATFORM_SETTINGS_ID, CANONICAL_ENGINE]
    );

    await client.query(
      `
        UPDATE platform_settings
        SET
          confidence_threshold = 0.91,
          updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
          AND confidence_threshold = 0.82
      `,
      [PLATFORM_SETTINGS_ID]
    );

    await client.query(
      `
        UPDATE platform_settings
        SET
          active_engine = $2,
          updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
          AND LOWER(active_engine) IN ('opencv', 'face_recognition', 'deepface', 'insightface', 'arcface')
      `,
      [PLATFORM_SETTINGS_ID, CANONICAL_ENGINE]
    );

    await client.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM information_schema.tables
          WHERE table_schema = 'public'
            AND table_name = 'face_embeddings'
        ) THEN
          UPDATE face_embeddings
          SET engine = '${CANONICAL_ENGINE}'
          WHERE LOWER(engine) IN ('opencv', 'face_recognition', 'deepface', 'insightface', 'arcface');
        END IF;
      END $$
    `);

    await client.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM information_schema.tables
          WHERE table_schema = 'public'
            AND table_name = 'attendance_sessions'
        ) THEN
          UPDATE attendance_sessions
          SET engine = '${CANONICAL_ENGINE}'
          WHERE LOWER(engine) IN ('opencv', 'face_recognition', 'deepface', 'insightface', 'arcface');
        END IF;
      END $$
    `);

  } finally {
    client.release();
  }
}

export async function ensureDatabaseReady() {
  if (!globalThis[DB_INIT_KEY]) {
    globalThis[DB_INIT_KEY] = initializeDatabase().catch((error) => {
      globalThis[DB_INIT_KEY] = null;
      throw error;
    });
  }

  return globalThis[DB_INIT_KEY];
}

export async function runQuery(text, params = [], runner = null) {
  if (runner) {
    return runner.query(text, params);
  }

  await ensureDatabaseReady();
  return getPool().query(text, params);
}

export async function runInTransaction(callback) {
  await ensureDatabaseReady();

  const client = await getPool().connect();

  try {
    await client.query("BEGIN");
    const result = await callback(client);
    await client.query("COMMIT");
    return result;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}

export function createId(prefix) {
  return `${prefix}-${randomUUID().replace(/-/g, "").slice(0, 16)}`;
}

export function serializeDate(value) {
  if (!value) {
    return null;
  }

  const date = value instanceof Date ? value : new Date(value);

  if (Number.isNaN(date.getTime())) {
    return null;
  }

  return date.toISOString();
}

export function mapPlatformSettings(row) {
  return {
    activeEngine: normalizeEngineName(row.active_engine),
    confidenceThreshold: Number(row.confidence_threshold),
    duplicateWindowMinutes: Number(row.duplicate_window_minutes),
    livenessEnabled: Boolean(row.liveness_enabled),
    antiSpoofMode: row.anti_spoof_mode,
    autoScanIntervalMs: Number(row.auto_scan_interval_ms),
    defaultCamera: row.default_camera,
    cameraEnabled: Boolean(row.camera_enabled),
    notificationsEnabled: Boolean(row.notifications_enabled)
  };
}

export async function getPlatformSettings(runner = null) {
  const result = await runQuery(
    `
      SELECT
        id,
        active_engine,
        confidence_threshold,
        duplicate_window_minutes,
        liveness_enabled,
        anti_spoof_mode,
        auto_scan_interval_ms,
        default_camera,
        camera_enabled,
        notifications_enabled
      FROM platform_settings
      WHERE id = $1
      LIMIT 1
    `,
    [PLATFORM_SETTINGS_ID],
    runner
  );

  return mapPlatformSettings(result.rows[0]);
}

export async function updatePlatformSettings(fields, runner) {
  const columnMap = {
    activeEngine: "active_engine",
    confidenceThreshold: "confidence_threshold",
    duplicateWindowMinutes: "duplicate_window_minutes",
    livenessEnabled: "liveness_enabled",
    antiSpoofMode: "anti_spoof_mode",
    autoScanIntervalMs: "auto_scan_interval_ms",
    defaultCamera: "default_camera",
    cameraEnabled: "camera_enabled",
    notificationsEnabled: "notifications_enabled"
  };

  const entries = Object.entries(fields).filter(
    ([key, value]) => columnMap[key] && value !== undefined
  );

  if (!entries.length) {
    return getPlatformSettings(runner);
  }

  const assignments = entries.map(
    ([key], index) => `${columnMap[key]} = $${index + 1}`
  );
  const values = entries.map(([key, value]) =>
    key === "activeEngine" ? normalizeEngineName(value) : value
  );

  const result = await runQuery(
    `
      UPDATE platform_settings
      SET
        ${assignments.join(", ")},
        updated_at = CURRENT_TIMESTAMP
      WHERE id = $${entries.length + 1}
      RETURNING
        id,
        active_engine,
        confidence_threshold,
        duplicate_window_minutes,
        liveness_enabled,
        anti_spoof_mode,
        auto_scan_interval_ms,
        default_camera,
        camera_enabled,
        notifications_enabled
    `,
    [...values, PLATFORM_SETTINGS_ID],
    runner
  );

  return mapPlatformSettings(result.rows[0]);
}
