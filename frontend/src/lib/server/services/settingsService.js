import { getAiServiceUrl, getDatabaseStatus } from "@/lib/server/dataAccess";
import {
  getPlatformSettings,
  runInTransaction,
  runQuery,
  updatePlatformSettings
} from "@/lib/server/db";
import { listAuditLogs, logAudit } from "@/lib/server/services/auditService";

const ALLOWED_SETTINGS = [
  "activeEngine",
  "confidenceThreshold",
  "duplicateWindowMinutes",
  "livenessEnabled",
  "antiSpoofMode",
  "autoScanIntervalMs",
  "defaultCamera",
  "cameraEnabled",
  "notificationsEnabled"
];

export async function getSettings() {
  const [settings, studentCountResult, auditCountResult, auditLogs] = await Promise.all([
    getPlatformSettings(),
    runQuery(`
      SELECT
        COUNT(*) FILTER (WHERE face_enrollment_status = 'completed') AS enrolled_students,
        COUNT(*) AS total_students
      FROM students
    `),
    runQuery(`SELECT COUNT(*) AS audit_entries FROM audit_logs`),
    listAuditLogs(12)
  ]);

  const counts = studentCountResult.rows[0];
  const auditCount = auditCountResult.rows[0];
  const databaseStatus = getDatabaseStatus();

  return {
    ...settings,
    health: {
      databaseMode: databaseStatus.mode,
      aiServiceUrl: getAiServiceUrl(),
      enrolledStudents: Number(counts.enrolled_students || 0),
      totalStudents: Number(counts.total_students || 0),
      auditEntries: Number(auditCount.audit_entries || 0)
    },
    auditLogs
  };
}

export async function updateSettings(payload, actor = {}) {
  const fields = {};

  ALLOWED_SETTINGS.forEach((field) => {
    if (payload[field] !== undefined) {
      fields[field] = payload[field];
    }
  });

  await runInTransaction(async (client) => {
    await updatePlatformSettings(fields, client);
    await logAudit(
      {
        actorId: actor.id || "system",
        actorName: actor.name || "System",
        action: "SETTINGS_UPDATED",
        entityType: "settings",
        entityId: "platform",
        message: "Core attendance configuration updated."
      },
      client
    );
  });

  return getSettings();
}
