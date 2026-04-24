import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import { getCameraStatus } from "@/lib/server/services/attendanceService";
import { getAiHealth } from "@/lib/server/services/cameraBridgeService";

export default async function handler(req, res) {
  if (req.method !== "GET") {
    return methodNotAllowed(res, ["GET"]);
  }

  try {
    const [cameraStatus, aiHealth] = await Promise.all([
      getCameraStatus(),
      getAiHealth()
    ]);

    return sendSuccess(res, {
      ...cameraStatus,
      aiHealth
    });
  } catch (error) {
    return sendError(res, error);
  }
}
