export function sendSuccess(res, data, statusCode = 200) {
  return res.status(statusCode).json({
    success: true,
    data
  });
}

export function sendError(res, error, statusCode = 500) {
  const message =
    typeof error === "string"
      ? error
      : error?.message || "Unexpected server error occurred.";

  return res.status(statusCode).json({
    success: false,
    message
  });
}

export function methodNotAllowed(res, allowedMethods = []) {
  res.setHeader("Allow", allowedMethods);
  return sendError(res, `Method ${res.req.method} not allowed.`, 405);
}
