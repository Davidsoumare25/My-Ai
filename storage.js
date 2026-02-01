export function saveResult(result) {
  localStorage.setItem("disciplineData", JSON.stringify(result));
}

export function getLastResult() {
  const data = localStorage.getItem("disciplineData");
  return data ? JSON.parse(data) : null;
}
