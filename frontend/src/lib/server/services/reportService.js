import { buildNotificationSeverity } from "@/lib/server/helpers";
import { runQuery } from "@/lib/server/db";
import { listAttendance, listSessions } from "@/lib/server/services/attendanceService";
import { listAuditLogs } from "@/lib/server/services/auditService";

function buildDailyTrend(records, days) {
  const series = [];
  const bucket = new Map();

  records.forEach((record) => {
    const key = record.recognizedAt?.slice(0, 10);

    if (key) {
      bucket.set(key, (bucket.get(key) || 0) + 1);
    }
  });

  for (let index = days - 1; index >= 0; index -= 1) {
    const date = new Date();
    date.setDate(date.getDate() - index);
    const key = date.toISOString().slice(0, 10);

    series.push({
      date: key,
      present: bucket.get(key) || 0
    });
  }

  return series;
}

function buildMonthlyTrend(records) {
  const currentMonth = new Date().toISOString().slice(0, 7);
  const buckets = new Map();

  records.forEach((record) => {
    const key = record.recognizedAt?.slice(0, 10);

    if (key?.startsWith(currentMonth)) {
      buckets.set(key, (buckets.get(key) || 0) + 1);
    }
  });

  return Array.from(buckets.entries())
    .sort((left, right) => left[0].localeCompare(right[0]))
    .map(([date, present]) => ({ date, present }));
}

export async function getAnalytics(filters = {}) {
  const trendFilters = { ...filters };
  delete trendFilters.date;

  const [attendancePayload, trendPayload, sessionsPayload, auditLogs, departmentsResult, studentCountResult] =
    await Promise.all([
      listAttendance(filters),
      listAttendance(trendFilters),
      listSessions(),
      listAuditLogs(8),
      runQuery(
        `
          SELECT
            d.id,
            d.code,
            d.name,
            COUNT(s.id) AS student_count
          FROM departments d
          LEFT JOIN students s ON s.department_id = d.id
          GROUP BY d.id, d.code, d.name
          ORDER BY d.name ASC
        `
      ),
      runQuery(`SELECT COUNT(*) AS total_students FROM students`)
    ]);

  const allRecords = attendancePayload.rows;
  const trendRecords = trendPayload.rows;
  const todayKey = new Date().toISOString().slice(0, 10);
  const todayCount = allRecords.filter((record) => record.recognizedAt?.startsWith(todayKey)).length;
  const duplicateBlocked = sessionsPayload.rows.reduce(
    (total, session) => total + session.duplicateCount,
    0
  );
  const unknownRejected = sessionsPayload.rows.reduce(
    (total, session) => total + session.unknownCount + session.lowConfidenceCount,
    0
  );
  const totalStudents = Number(studentCountResult.rows[0].total_students || 0);

  return {
    cards: [
      {
        title: "Registered Students",
        value: totalStudents,
        subtitle: "Across all departments",
        tone: "primary"
      },
      {
        title: "Today Present",
        value: todayCount,
        subtitle: "Auto-marked via AI recognition",
        tone: "success"
      },
      {
        title: "Active Sessions",
        value: sessionsPayload.activeSession ? 1 : 0,
        subtitle: sessionsPayload.activeSession
          ? sessionsPayload.activeSession.title
          : "No session running",
        tone: "secondary"
      },
      {
        title: "Duplicates Prevented",
        value: duplicateBlocked,
        subtitle: "Session-level duplicate blocking",
        tone: "warning"
      }
    ],
    summary: {
      ...attendancePayload.summary,
      unknownRejected
    },
    dailyTrend: buildDailyTrend(trendRecords, 7),
    monthlyTrend: buildMonthlyTrend(trendRecords),
    departmentDistribution: departmentsResult.rows.map((department) => ({
      department: department.code,
      students: Number(department.student_count || 0),
      present: trendRecords.filter((record) => record.departmentId === department.id).length
    })),
    recentAttendance: allRecords.slice(0, 8),
    sessions: sessionsPayload.rows.slice(0, 6),
    notifications: auditLogs.slice(0, 5).map((log) => ({
      id: log.id,
      title: log.action.replace(/_/g, " "),
      message: log.message,
      severity: buildNotificationSeverity(log.action),
      createdAt: log.createdAt
    })),
    auditLogs
  };
}
