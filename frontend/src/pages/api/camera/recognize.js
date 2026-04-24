import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import { recognizeFrame } from "@/lib/server/services/cameraBridgeService";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return methodNotAllowed(res, ["POST"]);
  }

  try {
    const { actor, ...payload } = req.body;
    const result = await recognizeFrame(payload, actor);
    return sendSuccess(res, result);
  } catch (error) {
    return sendError(res, error, 400);
  }
}
