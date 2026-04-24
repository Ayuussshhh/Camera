import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import { getStudentById, updateStudent } from "@/lib/server/services/studentService";

export default async function handler(req, res) {
  if (req.method === "GET") {
    try {
      const student = await getStudentById(req.query.id);
      return sendSuccess(res, student);
    } catch (error) {
      return sendError(res, error, 404);
    }
  }

  if (req.method === "PUT") {
    try {
      const { actor, ...payload } = req.body;
      const student = await updateStudent(req.query.id, payload, actor);
      return sendSuccess(res, student);
    } catch (error) {
      return sendError(res, error, 400);
    }
  }

  return methodNotAllowed(res, ["GET", "PUT"]);
}
