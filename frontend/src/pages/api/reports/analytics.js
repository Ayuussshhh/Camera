import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import { getAnalytics } from "@/lib/server/services/reportService";

export default async function handler(req, res) {
  if (req.method !== "GET") {
    return methodNotAllowed(res, ["GET"]);
  }

  try {
    const payload = await getAnalytics(req.query);
    return sendSuccess(res, payload);
  } catch (error) {
    return sendError(res, error);
  }
}
