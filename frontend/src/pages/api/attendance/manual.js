import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import { markManualAttendance } from "@/lib/server/services/attendanceService";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return methodNotAllowed(res, ["POST"]);
  }

  try {
    const { actor, ...payload } = req.body;
    const attendance = await markManualAttendance(payload, actor);
    return sendSuccess(res, attendance);
  } catch (error) {
    return sendError(res, error, 400);
  }
}
