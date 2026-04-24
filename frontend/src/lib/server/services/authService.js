import { runQuery } from "@/lib/server/db";
import { logAudit } from "@/lib/server/services/auditService";

function buildAuthUser(row) {
  return {
    id: row.id,
    name: row.name,
    email: row.email,
    role: row.role,
    teacherId: row.teacher_id || null,
    departmentId: row.department_id || null
  };
}

export async function login(credentials) {
  const email = credentials.email?.trim().toLowerCase();

  if (!email || !credentials.password || !credentials.role) {
    throw new Error("Email, password, and role are required.");
  }

  const result = await runQuery(
    `
      SELECT
        u.id,
        u.name,
        u.email,
        u.role,
        u.password_hash,
        t.id AS teacher_id,
        t.department_id
      FROM users u
      LEFT JOIN teachers t ON t.user_id = u.id
      WHERE LOWER(u.email) = $1
        AND u.role = $2
        AND u.is_active = TRUE
      LIMIT 1
    `,
    [email, credentials.role]
  );

  const user = result.rows[0];

  if (!user || user.password_hash !== credentials.password) {
    throw new Error("Invalid login credentials.");
  }

  const authenticatedUser = buildAuthUser(user);

  await logAudit({
    actorId: authenticatedUser.id,
    actorName: authenticatedUser.name,
    action: "LOGIN_SUCCESS",
    entityType: "user",
    entityId: authenticatedUser.id,
    message: `${authenticatedUser.role} login completed successfully.`
  });

  return authenticatedUser;
}
