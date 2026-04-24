import dayjs from "dayjs";

export function formatDateTime(value) {
  if (!value) {
    return "-";
  }

  return dayjs(value).format("DD MMM YYYY, hh:mm A");
}

export function formatDate(value) {
  if (!value) {
    return "-";
  }

  return dayjs(value).format("DD MMM YYYY");
}

export function formatTime(value) {
  if (!value) {
    return "-";
  }

  return dayjs(value).format("hh:mm A");
}

export function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "0%";
  }

  return `${Math.round(Number(value) * 100)}%`;
}

export function formatLabel(value) {
  if (!value) {
    return "-";
  }

  return value
    .toString()
    .replace(/[_-]/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
