import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import { stopSession } from "@/lib/server/services/attendanceService";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return methodNotAllowed(res, ["POST"]);
  }

  try {
    const payload = await stopSession(req.body.sessionId, req.body.actor);
    return sendSuccess(res, payload);
  } catch (error) {
    return sendError(res, error, 400);
  }
}
