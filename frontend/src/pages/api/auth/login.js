import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import { login } from "@/lib/server/services/authService";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return methodNotAllowed(res, ["POST"]);
  }

  try {
    const user = await login(req.body);
    return sendSuccess(res, user);
  } catch (error) {
    return sendError(res, error, 401);
  }
}
