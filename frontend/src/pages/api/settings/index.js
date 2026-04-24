import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import {
  getSettings,
  updateSettings
} from "@/lib/server/services/settingsService";

export default async function handler(req, res) {
  if (req.method === "GET") {
    try {
      const payload = await getSettings();
      return sendSuccess(res, payload);
    } catch (error) {
      return sendError(res, error);
    }
  }

  if (req.method === "PUT") {
    try {
      const { actor, ...payload } = req.body;
      const settings = await updateSettings(payload, actor);
      return sendSuccess(res, settings);
    } catch (error) {
      return sendError(res, error, 400);
    }
  }

  return methodNotAllowed(res, ["GET", "PUT"]);
}
