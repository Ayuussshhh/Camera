import { methodNotAllowed, sendError, sendSuccess } from "@/lib/server/apiResponses";
import { registerStudentWithAi } from "@/lib/server/services/cameraBridgeService";
import { enrollStudentFace } from "@/lib/server/services/studentService";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return methodNotAllowed(res, ["POST"]);
  }

  try {
    const { actor, studentId, images, engine } = req.body;
    const aiRegistration = await registerStudentWithAi({
      studentId,
      images,
      engine
    });

    const student = await enrollStudentFace(
      {
        studentId,
        engine: aiRegistration.engine || engine,
        imageCount: aiRegistration.imageCount || images?.length || 0,
        embeddingPath: aiRegistration.embeddingPath,
        lastTrainedAt: aiRegistration.lastTrainedAt
      },
      actor
    );

    return sendSuccess(res, {
      student,
      aiRegistration
    });
  } catch (error) {
    return sendError(res, error, 400);
  }
}
