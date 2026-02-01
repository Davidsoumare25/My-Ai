export function analyzeDiscipline(data) {
  let score = 0;

  score += Math.min(data.study * 15, 30);
  score += Math.min(data.sleep * 3, 20);
  score += data.phone <= 3 ? 20 : 5;
  score += data.goal ? 30 : 0;

  let level;
  let advice;

  if (score < 40) {
    level = "Faible";
    advice = "Commence petit. Réduis le téléphone et fixe un objectif simple.";
  } else if (score < 70) {
    level = "Moyen";
    advice = "Bonne base. Sois plus constant chaque jour.";
  } else if (score < 90) {
    level = "Fort";
    advice = "Très bonne discipline. Continue sans relâcher.";
  } else {
    level = "Élite";
    advice = "Excellente maîtrise. Tu es un modèle de rigueur.";
  }

  return { score, level, advice };
}
