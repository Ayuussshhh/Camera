export function normalizeSearchValue(value) {
  return value?.toString().trim().toLowerCase() || "";
}

export function matchesSearch(fields, search) {
  if (!search) {
    return true;
  }

  const query = normalizeSearchValue(search);

  return fields.some((field) => normalizeSearchValue(field).includes(query));
}

export function isSameDay(left, right) {
  if (!left || !right) {
    return false;
  }

  return new Date(left).toISOString().slice(0, 10) === right;
}

export function sortByNewest(collection, key) {
  return [...collection].sort(
    (firstItem, secondItem) =>
      new Date(secondItem[key]).getTime() - new Date(firstItem[key]).getTime()
  );
}

export function buildNotificationSeverity(action) {
  if (action.includes("FAILED") || action.includes("REJECTED")) {
    return "warning";
  }

  if (action.includes("UNKNOWN")) {
    return "info";
  }

  return "success";
}
