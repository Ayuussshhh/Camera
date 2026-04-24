import { createId, runQuery, serializeDate } from "@/lib/server/db";

function mapAuditRecord(row) {
  return {
    id: row.id,
    actorId: row.actor_id,
    actorName: row.actor_name,
    action: row.action,
    entityType: row.entity_type,
    entityId: row.entity_id,
    message: row.message,
    createdAt: serializeDate(row.created_at)
  };
}

export async function logAudit(
  {
    actorId = "system",
    actorName = "System",
    action,
    entityType,
    entityId,
    message
  },
  runner = null
) {
  const result = await runQuery(
    `
      INSERT INTO audit_logs (
        id,
        actor_id,
        actor_name,
        action,
        entity_type,
        entity_id,
        message
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7)
      RETURNING
        id,
        actor_id,
        actor_name,
        action,
        entity_type,
        entity_id,
        message,
        created_at
    `,
    [createId("audit"), actorId, actorName, action, entityType, entityId || null, message],
    runner
  );

  return mapAuditRecord(result.rows[0]);
}

export async function listAuditLogs(limit = 12) {
  const result = await runQuery(
    `
      SELECT
        id,
        actor_id,
        actor_name,
        action,
        entity_type,
        entity_id,
        message,
        created_at
      FROM audit_logs
      ORDER BY created_at DESC
      LIMIT $1
    `,
    [limit]
  );

  return result.rows.map(mapAuditRecord);
}
