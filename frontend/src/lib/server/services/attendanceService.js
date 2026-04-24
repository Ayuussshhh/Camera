import {
  createId,
  getPlatformSettings,
  runInTransaction,
  runQuery,
  serializeDate
} from "@/lib/server/db";
import { logAudit } from "@/lib/server/services/auditService";
import { startAiSession, stopAiSession } from "@/lib/server/aiServiceClient";
import { normalizeEngineName } from "@/utils/engine";

const DECORATED_SESSION_SELECT = `
  SELECT
    s.id,
    s.title,
    s.subject,
    s.department_id,
    d.name AS department_name,
    d.code AS department_code,
    s.semester,
    s.section,
    s.teacher_id,
    COALESCE(t.full_name, '-') AS teacher_name,
    s.engine,
    s.status,
    s.started_at,
    s.ended_at,
    s.recognized_count,
    s.duplicate_count,
    s.unknown_count,
    s.low_confidence_count
  FROM attendance_sessions s
  INNER JOIN departments d ON d.id = s.department_id
  LEFT JOIN teachers t ON t.id = s.teacher_id
`;

function mapSessionRow(row) {
  return {
    id: row.id,
    title: row.title,
    subject: row.subject,
    departmentId: row.department_id,
    departmentName: row.department_name,
    departmentCode: row.department_code,
    semester: row.semester,
    section: row.section,
    teacherId: row.teacher_id,
    teacherName: row.teacher_name || "-",
    engine: normalizeEngineName(row.engine),
    status: row.status,
    startedAt: serializeDate(row.started_at),
    endedAt: serializeDate(row.ended_at),
    recognizedCount: Number(row.recognized_count || 0),
    duplicateCount: Number(row.duplicate_count || 0),
    unknownCount: Number(row.unknown_count || 0),
    lowConfidenceCount: Number(row.low_confidence_count || 0)
  };
}

function mapAttendanceRow(row) {
  return {
    id: row.id,
    sessionId: row.session_id,
    studentId: row.student_id,
    status: row.status,
    confidence: Number(row.confidence || 0),
    source: row.source,
    recognizedAt: serializeDate(row.recognized_at),
    studentName: row.student_name,
    rollNumber: row.roll_number,
    departmentId: row.department_id,
    departmentName: row.department_name,
    departmentCode: row.department_code,
    semester: row.semester,
    section: row.section,
    subject: row.subject,
    sessionTitle: row.session_title
  };
}

async function findTeacherId(actor, runner) {
  if (actor.teacherId) {
    return actor.teacherId;
  }

  if (!actor.email) {
    return null;
  }

  const result = await runQuery(
    `
      SELECT id
      FROM teachers
      WHERE LOWER(email) = LOWER($1)
      LIMIT 1
    `,
    [actor.email],
    runner
  );

  return result.rows[0]?.id || null;
}

async function getSessionById(sessionId, runner = null) {
  const result = await runQuery(
    `
      ${DECORATED_SESSION_SELECT}
      WHERE s.id = $1
      LIMIT 1
    `,
    [sessionId],
    runner
  );

  return result.rows[0] ? mapSessionRow(result.rows[0]) : null;
}

async function getAttendanceById(attendanceId, runner = null) {
  const result = await runQuery(
    `
      SELECT
        a.id,
        a.session_id,
        a.student_id,
        a.status,
        a.confidence,
        a.source,
        a.recognized_at,
        st.full_name AS student_name,
        st.roll_number,
        st.department_id,
        d.name AS department_name,
        d.code AS department_code,
        st.semester,
        st.section,
        s.subject,
        s.title AS session_title
      FROM attendance a
      INNER JOIN students st ON st.id = a.student_id
      INNER JOIN attendance_sessions s ON s.id = a.session_id
      INNER JOIN departments d ON d.id = st.department_id
      WHERE a.id = $1
      LIMIT 1
    `,
    [attendanceId],
    runner
  );

  return result.rows[0] ? mapAttendanceRow(result.rows[0]) : null;
}

export async function getActiveSession() {
  const result = await runQuery(
    `
      ${DECORATED_SESSION_SELECT}
      WHERE s.status = 'active'
      ORDER BY s.started_at DESC
      LIMIT 1
    `
  );

  return result.rows[0] ? mapSessionRow(result.rows[0]) : null;
}

export async function listSessions() {
  const result = await runQuery(
    `
      ${DECORATED_SESSION_SELECT}
      ORDER BY s.started_at DESC
    `
  );

  const rows = result.rows.map(mapSessionRow);

  return {
    rows,
    activeSession: rows.find((session) => session.status === "active") || null
  };
}

export async function listAttendance(filters = {}) {
  const conditions = [];
  const params = [];

  if (filters.date) {
    params.push(filters.date);
    conditions.push(`DATE(a.recognized_at) = $${params.length}`);
  }

  if (filters.departmentId) {
    params.push(filters.departmentId);
    conditions.push(`st.department_id = $${params.length}`);
  }

  if (filters.semester) {
    params.push(filters.semester);
    conditions.push(`st.semester = $${params.length}`);
  }

  if (filters.section) {
    params.push(filters.section);
    conditions.push(`st.section = $${params.length}`);
  }

  const recordsResult = await runQuery(
    `
      SELECT
        a.id,
        a.session_id,
        a.student_id,
        a.status,
        a.confidence,
        a.source,
        a.recognized_at,
        st.full_name AS student_name,
        st.roll_number,
        st.department_id,
        d.name AS department_name,
        d.code AS department_code,
        st.semester,
        st.section,
        s.subject,
        s.title AS session_title
      FROM attendance a
      INNER JOIN students st ON st.id = a.student_id
      INNER JOIN attendance_sessions s ON s.id = a.session_id
      INNER JOIN departments d ON d.id = st.department_id
      ${conditions.length ? `WHERE ${conditions.join(" AND ")}` : ""}
      ORDER BY a.recognized_at DESC
    `,
    params
  );

  const rosterConditions = [];
  const rosterParams = [];

  if (filters.departmentId) {
    rosterParams.push(filters.departmentId);
    rosterConditions.push(`department_id = $${rosterParams.length}`);
  }

  if (filters.semester) {
    rosterParams.push(filters.semester);
    rosterConditions.push(`semester = $${rosterParams.length}`);
  }

  if (filters.section) {
    rosterParams.push(filters.section);
    rosterConditions.push(`section = $${rosterParams.length}`);
  }

  const rosterResult = await runQuery(
    `
      SELECT COUNT(*) AS total_students
      FROM students
      ${rosterConditions.length ? `WHERE ${rosterConditions.join(" AND ")}` : ""}
    `,
    rosterParams
  );

  const rows = recordsResult.rows.map(mapAttendanceRow);
  const presentCount = new Set(rows.map((record) => record.studentId)).size;
  const totalStudents = Number(rosterResult.rows[0].total_students || 0);

  return {
    rows,
    summary: {
      totalStudents,
      presentCount,
      absenceCount: Math.max(totalStudents - presentCount, 0),
      presenceRate: totalStudents ? presentCount / totalStudents : 0
    }
  };
}

export async function startSession(payload, actor = {}) {
  if (!payload.departmentId || !payload.semester || !payload.section || !payload.subject) {
    throw new Error("Department, semester, section, and subject are required to start a session.");
  }

  const settings = await getPlatformSettings();
  const sessionId = createId("sess");
  const teacherId = await findTeacherId(actor);
  const sessionData = {
    id: sessionId,
    title: payload.title || `${payload.subject} Attendance Session`,
    subject: payload.subject,
    departmentId: payload.departmentId,
    semester: payload.semester,
    section: payload.section,
    teacherId,
    engine: normalizeEngineName(payload.engine || settings.activeEngine)
  };

  await startAiSession({
    sessionId: sessionData.id,
    engine: sessionData.engine,
    threshold: settings.confidenceThreshold,
    metadata: {
      title: sessionData.title,
      subject: sessionData.subject,
      departmentId: sessionData.departmentId,
      semester: sessionData.semester,
      section: sessionData.section
    }
  });

  try {
    await runInTransaction(async (client) => {
      const activeSessionResult = await runQuery(
        `
          SELECT id
          FROM attendance_sessions
          WHERE status = 'active'
          LIMIT 1
          FOR UPDATE
        `,
        [],
        client
      );

      if (activeSessionResult.rows[0]) {
        throw new Error("An attendance session is already active.");
      }

      await runQuery(
        `
          INSERT INTO attendance_sessions (
            id,
            title,
            subject,
            department_id,
            semester,
            section,
            teacher_id,
            engine,
            status
          )
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'active')
        `,
        [
          sessionData.id,
          sessionData.title,
          sessionData.subject,
          sessionData.departmentId,
          sessionData.semester,
          sessionData.section,
          sessionData.teacherId,
          sessionData.engine
        ],
        client
      );

      await logAudit(
        {
          actorId: actor.id || "system",
          actorName: actor.name || "System",
          action: "SESSION_STARTED",
          entityType: "attendance_session",
          entityId: sessionData.id,
          message: `${sessionData.title} started for ${sessionData.departmentId}.`
        },
        client
      );
    });
  } catch (error) {
    await stopAiSession(sessionData.id).catch(() => undefined);
    throw error;
  }

  return getSessionById(sessionData.id);
}

export async function stopSession(sessionId, actor = {}) {
  const session = await runInTransaction(async (client) => {
    const existingResult = await runQuery(
      `
        SELECT id, status, title
        FROM attendance_sessions
        WHERE id = $1
        LIMIT 1
        FOR UPDATE
      `,
      [sessionId],
      client
    );

    const existing = existingResult.rows[0];

    if (!existing) {
      throw new Error("Attendance session not found.");
    }

    if (existing.status !== "active") {
      throw new Error("Only active attendance sessions can be stopped.");
    }

    await runQuery(
      `
        UPDATE attendance_sessions
        SET
          status = 'completed',
          ended_at = CURRENT_TIMESTAMP
        WHERE id = $1
      `,
      [sessionId],
      client
    );

    await logAudit(
      {
        actorId: actor.id || "system",
        actorName: actor.name || "System",
        action: "SESSION_COMPLETED",
        entityType: "attendance_session",
        entityId: sessionId,
        message: `${existing.title} stopped by ${actor.name || "System"}.`
      },
      client
    );

    return getSessionById(sessionId, client);
  });

  await stopAiSession(sessionId).catch(() => undefined);
  return session;
}

export async function markAttendance(payload, actor = {}) {
  return runInTransaction(async (client) => {
    const settings = await getPlatformSettings(client);
    const sessionResult = await runQuery(
      `
        SELECT
          id,
          title,
          subject,
          department_id,
          semester,
          section,
          status
        FROM attendance_sessions
        WHERE id = $1
        LIMIT 1
        FOR UPDATE
      `,
      [payload.sessionId],
      client
    );

    const session = sessionResult.rows[0];

    if (!session || (session.status !== "active" && session.status !== "completed")) {
      throw new Error("Active attendance session is required.");
    }

    if (!payload.studentId) {
      await runQuery(
        `
          UPDATE attendance_sessions
          SET unknown_count = unknown_count + 1
          WHERE id = $1
        `,
        [session.id],
        client
      );

      await logAudit(
        {
          actorId: actor.id || "system",
          actorName: actor.name || "System",
          action: "UNKNOWN_FACE_IGNORED",
          entityType: "attendance_session",
          entityId: session.id,
          message: "Unknown face ignored during recognition."
        },
        client
      );

      return {
        status: "unknown",
        message: "Unknown face ignored."
      };
    }

    if (Number(payload.confidence || 0) < settings.confidenceThreshold) {
      await runQuery(
        `
          UPDATE attendance_sessions
          SET low_confidence_count = low_confidence_count + 1
          WHERE id = $1
        `,
        [session.id],
        client
      );

      await logAudit(
        {
          actorId: actor.id || "system",
          actorName: actor.name || "System",
          action: "LOW_CONFIDENCE_REJECTED",
          entityType: "attendance_session",
          entityId: session.id,
          message: `Face rejected below threshold for ${payload.studentId}.`
        },
        client
      );

      return {
        status: "rejected",
        message: "Recognition confidence below threshold."
      };
    }

    const studentResult = await runQuery(
      `
        SELECT
          id,
          full_name,
          roll_number,
          department_id,
          semester,
          section,
          face_enrollment_status
        FROM students
        WHERE id = $1
        LIMIT 1
      `,
      [payload.studentId],
      client
    );

    const student = studentResult.rows[0];

    if (!student || student.face_enrollment_status !== "completed") {
      await runQuery(
        `
          UPDATE attendance_sessions
          SET unknown_count = unknown_count + 1
          WHERE id = $1
        `,
        [session.id],
        client
      );

      return {
        status: "unknown",
        message: "Student is not enrolled for face recognition."
      };
    }

    const studentMatchesSession =
      student.department_id === session.department_id &&
      student.semester === session.semester &&
      student.section === session.section;

    if (!studentMatchesSession) {
      return {
        status: "rejected",
        message: "Student does not belong to the active academic group."
      };
    }

    try {
      const attendanceResult = await runQuery(
        `
          INSERT INTO attendance (
            id,
            session_id,
            student_id,
            status,
            confidence,
            source
          )
          VALUES ($1, $2, $3, 'present', $4, $5)
          RETURNING id
        `,
        [
          createId("att"),
          session.id,
          student.id,
          Number(payload.confidence || 0),
          payload.source || "ai-camera"
        ],
        client
      );

      await runQuery(
        `
          UPDATE attendance_sessions
          SET recognized_count = recognized_count + 1
          WHERE id = $1
        `,
        [session.id],
        client
      );

      await logAudit(
        {
          actorId: actor.id || "system",
          actorName: actor.name || "System",
          action: "ATTENDANCE_MARKED",
          entityType: "attendance",
          entityId: attendanceResult.rows[0].id,
          message: `${student.full_name} marked present automatically.`
        },
        client
      );

      return {
        status: "marked",
        message: "Attendance marked successfully.",
        attendance: await getAttendanceById(attendanceResult.rows[0].id, client)
      };
    } catch (error) {
      if (error.code !== "23505") {
        throw error;
      }

      await runQuery(
        `
          UPDATE attendance_sessions
          SET duplicate_count = duplicate_count + 1
          WHERE id = $1
        `,
        [session.id],
        client
      );

      const existingAttendanceResult = await runQuery(
        `
          SELECT id
          FROM attendance
          WHERE session_id = $1
            AND student_id = $2
          LIMIT 1
        `,
        [session.id, student.id],
        client
      );

      const existingAttendanceId = existingAttendanceResult.rows[0]?.id;

      await logAudit(
        {
          actorId: actor.id || "system",
          actorName: actor.name || "System",
          action: "DUPLICATE_BLOCKED",
          entityType: "attendance",
          entityId: existingAttendanceId || null,
          message: `Duplicate attendance blocked for ${student.full_name}.`
        },
        client
      );

      return {
        status: "duplicate",
        message: "Attendance already marked for this session.",
        attendance: existingAttendanceId
          ? await getAttendanceById(existingAttendanceId, client)
          : null
      };
    }
  });
}

export async function markManualAttendance(payload, actor = {}) {
  if (!payload.sessionId || !payload.studentId) {
    throw new Error("sessionId and studentId are required.");
  }

  return markAttendance(
    {
      sessionId: payload.sessionId,
      studentId: payload.studentId,
      confidence: payload.confidence || 1,
      source: payload.source || "manual-dashboard"
    },
    actor
  );
}

export async function getCameraStatus() {
  const [settings, activeSession] = await Promise.all([
    getPlatformSettings(),
    getActiveSession()
  ]);

  return {
    deviceName: settings.defaultCamera,
    online: settings.cameraEnabled,
    resolution: process.env.AI_CAMERA_RESOLUTION || "1280x720",
    fps: Number(process.env.AI_CAMERA_FPS || 24),
    lastHeartbeat: new Date().toISOString(),
    activeSession,
    threshold: settings.confidenceThreshold,
    engine: activeSession?.engine || settings.activeEngine,
    livenessEnabled: settings.livenessEnabled,
    duplicateWindowMinutes: settings.duplicateWindowMinutes,
    autoScanIntervalMs: settings.autoScanIntervalMs,
    cameraEnabled: settings.cameraEnabled
  };
}
