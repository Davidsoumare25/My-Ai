import { analyzeDiscipline } from "./ai.js";
import { saveResult } from "./storage.js";

document.getElementById("analyzeBtn").addEventListener("click", () => {
  const data = {
    study: Number(document.getElementById("study").value),
    sleep: Number(document.getElementById("sleep").value),
    phone: Number(document.getElementById("phone").value),
    goal: document.getElementById("goal").checked
  };

  const result = analyzeDiscipline(data);
  saveResult(result);

  document.getElementById("score").textContent = `Score : ${result.score}/100`;
  document.getElementById("level").textContent = `Niveau : ${result.level}`;
  document.getElementById("advice").textContent = result.advice;
});
