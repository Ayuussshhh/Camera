import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import { listAttendance } from "@/lib/server/services/attendanceService";

export default async function handler(req, res) {
  if (req.method !== "GET") {
    return methodNotAllowed(res, ["GET"]);
  }

  try {
    const payload = await listAttendance(req.query);
    return sendSuccess(res, payload);
  } catch (error) {
    return sendError(res, error);
  }
}
