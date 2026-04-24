import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import {
  listSessions,
  startSession
} from "@/lib/server/services/attendanceService";

export default async function handler(req, res) {
  if (req.method === "GET") {
    try {
      const payload = await listSessions();
      return sendSuccess(res, payload);
    } catch (error) {
      return sendError(res, error);
    }
  }

  if (req.method === "POST") {
    try {
      const { actor, ...payload } = req.body;
      const session = await startSession(payload, actor);
      return sendSuccess(res, session, 201);
    } catch (error) {
      return sendError(res, error, 400);
    }
  }

  return methodNotAllowed(res, ["GET", "POST"]);
}
