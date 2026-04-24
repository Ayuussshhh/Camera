import { createId, runInTransaction, runQuery, serializeDate } from "@/lib/server/db";
import { logAudit } from "@/lib/server/services/auditService";
import { getEngineLabel, normalizeEngineName } from "@/utils/engine";

function mapStudentRow(row) {
  return {
    id: row.id,
    userId: row.user_id,
    fullName: row.full_name,
    email: row.email,
    rollNumber: row.roll_number,
    departmentId: row.department_id,
    departmentName: row.department_name,
    departmentCode: row.department_code,
    semester: row.semester,
    section: row.section,
    guardianPhone: row.guardian_phone,
    faceEnrollmentStatus: row.face_enrollment_status,
    registeredAt: serializeDate(row.registered_at),
    engine: row.engine ? getEngineLabel(row.engine) : "-",
    faceImageCount: Number(row.face_image_count || 0),
    lastTrainedAt: serializeDate(row.last_trained_at)
  };
}

async function getDepartments(runner = null) {
  const result = await runQuery(
    `
      SELECT id, code, name
      FROM departments
      ORDER BY name ASC
    `,
    [],
    runner
  );

  return result.rows;
}

export async function listStudents(filters = {}) {
  const conditions = [];
  const params = [];

  if (filters.departmentId) {
    params.push(filters.departmentId);
    conditions.push(`s.department_id = $${params.length}`);
  }

  if (filters.semester) {
    params.push(filters.semester);
    conditions.push(`s.semester = $${params.length}`);
  }

  if (filters.section) {
    params.push(filters.section);
    conditions.push(`s.section = $${params.length}`);
  }

  if (filters.search?.trim()) {
    params.push(`%${filters.search.trim()}%`);
    conditions.push(
      `(s.full_name ILIKE $${params.length} OR s.roll_number ILIKE $${params.length} OR s.email ILIKE $${params.length})`
    );
  }

  const result = await runQuery(
    `
      SELECT
        s.id,
        s.user_id,
        s.full_name,
        s.email,
        s.roll_number,
        s.department_id,
        d.name AS department_name,
        d.code AS department_code,
        s.semester,
        s.section,
        s.guardian_phone,
        s.face_enrollment_status,
        s.registered_at,
        fe.engine,
        fe.image_count AS face_image_count,
        fe.last_trained_at
      FROM students s
      INNER JOIN departments d ON d.id = s.department_id
      LEFT JOIN LATERAL (
        SELECT
          engine,
          image_count,
          last_trained_at
        FROM face_embeddings
        WHERE student_id = s.id
        ORDER BY last_trained_at DESC NULLS LAST
        LIMIT 1
      ) fe ON TRUE
      ${conditions.length ? `WHERE ${conditions.join(" AND ")}` : ""}
      ORDER BY s.registered_at DESC
    `,
    params
  );

  return {
    rows: result.rows.map(mapStudentRow),
    departments: await getDepartments()
  };
}

export async function getStudentById(studentId, runner = null) {
  const result = await runQuery(
    `
      SELECT
        s.id,
        s.user_id,
        s.full_name,
        s.email,
        s.roll_number,
        s.department_id,
        d.name AS department_name,
        d.code AS department_code,
        s.semester,
        s.section,
        s.guardian_phone,
        s.face_enrollment_status,
        s.registered_at,
        fe.engine,
        fe.image_count AS face_image_count,
        fe.last_trained_at
      FROM students s
      INNER JOIN departments d ON d.id = s.department_id
      LEFT JOIN LATERAL (
        SELECT
          engine,
          image_count,
          last_trained_at
        FROM face_embeddings
        WHERE student_id = s.id
        ORDER BY last_trained_at DESC NULLS LAST
        LIMIT 1
      ) fe ON TRUE
      WHERE s.id = $1
      LIMIT 1
    `,
    [studentId],
    runner
  );

  const student = result.rows[0];

  if (!student) {
    throw new Error("Student record not found.");
  }

  return mapStudentRow(student);
}

export async function createStudent(payload, actor = {}) {
  const requiredFields = [
    "fullName",
    "email",
    "rollNumber",
    "departmentId",
    "semester",
    "section"
  ];

  const missingField = requiredFields.find((field) => !payload[field]);

  if (missingField) {
    throw new Error(`${missingField} is required.`);
  }

  const normalizedEmail = payload.email.trim().toLowerCase();

  return runInTransaction(async (client) => {
    const duplicateCheck = await runQuery(
      `
        SELECT
          EXISTS (
            SELECT 1
            FROM students
            WHERE LOWER(email) = $1
          ) AS email_exists,
          EXISTS (
            SELECT 1
            FROM students
            WHERE LOWER(roll_number) = LOWER($2)
          ) AS roll_number_exists
      `,
      [normalizedEmail, payload.rollNumber],
      client
    );

    if (duplicateCheck.rows[0].email_exists || duplicateCheck.rows[0].roll_number_exists) {
      throw new Error("Student with this email or roll number already exists.");
    }

    const userId = createId("user");
    const studentId = createId("stu");

    await runQuery(
      `
        INSERT INTO users (
          id,
          name,
          email,
          password_hash,
          role,
          is_active
        )
        VALUES ($1, $2, $3, $4, 'student', TRUE)
      `,
      [userId, payload.fullName, normalizedEmail, payload.password || "student123"],
      client
    );

    await runQuery(
      `
        INSERT INTO students (
          id,
          user_id,
          roll_number,
          full_name,
          email,
          department_id,
          semester,
          section,
          guardian_phone,
          face_enrollment_status
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'pending')
      `,
      [
        studentId,
        userId,
        payload.rollNumber,
        payload.fullName,
        normalizedEmail,
        payload.departmentId,
        payload.semester,
        payload.section,
        payload.guardianPhone || null
      ],
      client
    );

    await logAudit(
      {
        actorId: actor.id || "system",
        actorName: actor.name || "System",
        action: "STUDENT_CREATED",
        entityType: "student",
        entityId: studentId,
        message: `${payload.fullName} registered for attendance onboarding.`
      },
      client
    );

    return getStudentById(studentId, client);
  });
}

export async function updateStudent(studentId, payload, actor = {}) {
  if (!studentId) {
    throw new Error("studentId is required.");
  }

  const requiredFields = [
    "fullName",
    "email",
    "rollNumber",
    "departmentId",
    "semester",
    "section"
  ];

  const missingField = requiredFields.find((field) => !payload[field]);

  if (missingField) {
    throw new Error(`${missingField} is required.`);
  }

  const normalizedEmail = payload.email.trim().toLowerCase();

  return runInTransaction(async (client) => {
    const student = await getStudentById(studentId, client);

    const duplicateCheck = await runQuery(
      `
        SELECT
          EXISTS (
            SELECT 1
            FROM students
            WHERE LOWER(email) = $1
              AND id <> $2
          ) AS email_exists,
          EXISTS (
            SELECT 1
            FROM students
            WHERE LOWER(roll_number) = LOWER($3)
              AND id <> $2
          ) AS roll_number_exists
      `,
      [normalizedEmail, studentId, payload.rollNumber],
      client
    );

    if (duplicateCheck.rows[0].email_exists || duplicateCheck.rows[0].roll_number_exists) {
      throw new Error("Student with this email or roll number already exists.");
    }

    await runQuery(
      `
        UPDATE users
        SET
          name = $2,
          email = $3
        WHERE id = $1
      `,
      [student.userId, payload.fullName, normalizedEmail],
      client
    );

    await runQuery(
      `
        UPDATE students
        SET
          roll_number = $2,
          full_name = $3,
          email = $4,
          department_id = $5,
          semester = $6,
          section = $7,
          guardian_phone = $8
        WHERE id = $1
      `,
      [
        studentId,
        payload.rollNumber,
        payload.fullName,
        normalizedEmail,
        payload.departmentId,
        payload.semester,
        payload.section,
        payload.guardianPhone || null
      ],
      client
    );

    await logAudit(
      {
        actorId: actor.id || "system",
        actorName: actor.name || "System",
        action: "STUDENT_UPDATED",
        entityType: "student",
        entityId: studentId,
        message: `${payload.fullName} profile updated.`
      },
      client
    );

    return getStudentById(studentId, client);
  });
}

export async function enrollStudentFace(payload, actor = {}) {
  if (!payload.studentId || !payload.engine || !payload.embeddingPath) {
    throw new Error("studentId, engine, and embeddingPath are required.");
  }

  const normalizedEngine = normalizeEngineName(payload.engine);

  return runInTransaction(async (client) => {
    const student = await getStudentById(payload.studentId, client);

    await runQuery(
      `
        UPDATE face_embeddings
        SET engine = $2
        WHERE student_id = $1
          AND LOWER(engine) IN ('opencv', 'face_recognition', 'deepface', 'insightface', 'arcface')
      `,
      [payload.studentId, normalizedEngine],
      client
    );

    await runQuery(
      `
        INSERT INTO face_embeddings (
          id,
          student_id,
          engine,
          embedding_path,
          image_count,
          last_trained_at
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (student_id, engine)
        DO UPDATE SET
          embedding_path = EXCLUDED.embedding_path,
          image_count = EXCLUDED.image_count,
          last_trained_at = EXCLUDED.last_trained_at
      `,
      [
        createId("emb"),
        payload.studentId,
        normalizedEngine,
        payload.embeddingPath,
        Number(payload.imageCount || 0),
        payload.lastTrainedAt || new Date().toISOString()
      ],
      client
    );

    await runQuery(
      `
        UPDATE students
        SET face_enrollment_status = 'completed'
        WHERE id = $1
      `,
      [payload.studentId],
      client
    );

    await logAudit(
      {
        actorId: actor.id || "system",
        actorName: actor.name || "System",
        action: "FACE_ENROLLMENT_COMPLETED",
        entityType: "student",
        entityId: payload.studentId,
        message: `${student.fullName} face profile updated with ${Number(
          payload.imageCount || 0
        )} captures.`
      },
      client
    );

    return getStudentById(payload.studentId, client);
  });
}
