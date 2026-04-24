import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import {
  createStudent,
  listStudents
} from "@/lib/server/services/studentService";

export default async function handler(req, res) {
  if (req.method === "GET") {
    try {
      const payload = await listStudents(req.query);
      return sendSuccess(res, payload);
    } catch (error) {
      return sendError(res, error);
    }
  }

  if (req.method === "POST") {
    try {
      const { actor, ...payload } = req.body;
      const student = await createStudent(payload, actor);
      return sendSuccess(res, student, 201);
    } catch (error) {
      return sendError(res, error, 400);
    }
  }

  return methodNotAllowed(res, ["GET", "POST"]);
}
