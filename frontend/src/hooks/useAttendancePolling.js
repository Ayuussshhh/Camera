import { useEffect } from "react";

export default function useAttendancePolling(callback, enabled, intervalMs) {
  useEffect(() => {
    if (!enabled) {
      return undefined;
    }

    const timer = setInterval(callback, intervalMs);

    return () => clearInterval(timer);
  }, [callback, enabled, intervalMs]);
}
