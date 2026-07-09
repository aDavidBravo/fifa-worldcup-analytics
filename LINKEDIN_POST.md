# LinkedIn — playbook completo

Tres formas de mostrar el proyecto, de mayor a menor esfuerzo. Haz las tres:
la **1** para alcance hoy, la **2** para profundidad técnica, la **3** para que
quede permanente en tu perfil.

- **URLs:**
  - Dashboard: https://adavidbravo.github.io/fifa-worldcup-analytics/
  - Código: https://github.com/aDavidBravo/fifa-worldcup-analytics

---

## 1) Post en el feed (texto + imágenes) — publícalo YA, antes de semifinales

Adjunta 2-4 imágenes (ver sección "Capturas" abajo). La primera imagen es la que
engancha: usa la de **probabilidad de campeón con banderas**.

---

⚽ **¿Quién gana el Mundial 2026? Construí un modelo para averiguarlo — y luego lo puse a competir contra 3 técnicas de ML para ver si valía la pena.**

Mientras se juegan los cuartos de final, terminé un proyecto de data science de punta a punta: **96 años de historia del Mundial + un pronóstico en vivo del torneo que se juega ahora mismo**.

🔮 Mi modelo Elo + Monte Carlo (10.000 simulaciones) dice:
→ 🇦🇷 Argentina 28,9%
→ 🇪🇸 España 27,4%
→ 🇫🇷 Francia 19,7%
→ 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra 12,6%

Pero aquí está la parte que separa un gráfico bonito de trabajo de data scientist senior 👇

🧪 **No me quedé con el modelo simple porque sí. Lo comparé.**
Entrené 4 modelos sobre 19.673 partidos internacionales (2002-2022) y los evalué sobre 3.694 partidos que nunca vieron (2023-2026), con separación temporal (no un shuffle aleatorio, que filtraría el futuro al pasado):

| Modelo | Log-loss ↓ | Accuracy ↑ |
|---|---|---|
| Poisson Dixon-Coles | **0,8572** | 60,6% |
| XGBoost | 0,8573 | 60,5% |
| Red neuronal (MLP) | 0,8605 | 60,1% |
| Elo (baseline) | 0,8657 | **60,7%** |

El hallazgo: los modelos "avanzados" le ganan al Elo por **menos de 1%** de log-loss, y el Elo mantiene la mejor accuracy. Con solo ~64 partidos de Mundial de señal real, la sofisticación no compra casi nada — así que el pronóstico se queda en el modelo interpretable, **ahora validado, no asumido**.

Elegir el modelo simple *porque la evidencia lo respalda* es la decisión senior. 📊

**Stack:** Python (pandas/NumPy) · scikit-learn · XGBoost · SciPy (Dixon-Coles desde cero) · Apache ECharts. Cero backend, totalmente reproducible.

🔗 Dashboard interactivo: https://adavidbravo.github.io/fifa-worldcup-analytics/
💻 Código + README técnico: https://github.com/aDavidBravo/fifa-worldcup-analytics

¿Le ganará el modelo a los expertos? Lo sabremos el 19 de julio en el MetLife Stadium. 🏆

#DataScience #MachineLearning #Python #XGBoost #DataVisualization #SportsAnalytics #WorldCup2026

---

## 2) Post tipo "Documento" (carrusel PDF) — el formato para gráficos + detalle técnico

Este es el que pediste: profundidad técnica con gráficos. LinkedIn favorece el
formato documento (la gente desliza, sube el tiempo de permanencia). Arma un PDF
horizontal (1920×1080 por diapositiva) — te dejo el guion listo, diapositiva por
diapositiva. Puedes montarlo en Canva/PowerPoint/Google Slides en 30 min.

**Slide 1 — Portada (gancho)**
- Título: "¿Quién gana el Mundial 2026?"
- Subtítulo: "Un modelo de pronóstico — y cómo lo validé contra XGBoost, Poisson y una red neuronal"
- Fondo: captura del gráfico de probabilidad de campeón con banderas.

**Slide 2 — El problema**
- "Predecir un torneo de eliminación es difícil: pocos partidos, mucha varianza.
  ¿Se puede hacer bien? ¿Y hace falta deep learning?"

**Slide 3 — Los datos**
- 49.000+ partidos internacionales desde 1872 · 96 años de Mundiales (1930-2026)
- Fuentes: Fjelstul World Cup Database + martj42/international_results
- Captura: KPIs del dashboard (23 Mundiales · 1.060 partidos · 3.000 goles).

**Slide 4 — El modelo base: Elo + Monte Carlo**
- Explica Elo en una línea (rating que sube/baja según resultado esperado).
- Fórmula: P(A) = 1 / (1 + 10^(−ΔElo/400))
- 10.000 simulaciones del bracket → probabilidad de campeón.
- Captura: el bracket "camino a la final" con banderas.

**Slide 5 — "¿Pero por qué no deep learning?" (el giro)**
- "En vez de asumir, medí. Entrené 4 modelos con las mismas features y la misma
  validación temporal."
- Lista: Elo · Dixon-Coles Poisson (SciPy) · XGBoost · Red neuronal (MLP).

**Slide 6 — Feature engineering sin fuga temporal**
- Fragmento de código (el de features.py del README): "cada feature es la foto
  pre-partido; el estado se actualiza DESPUÉS de registrar la fila."
- Punto clave: "Un shuffle aleatorio filtraría forma futura al pasado. Split temporal."

**Slide 7 — Resultados (la tabla)**
- La tabla de log-loss / Brier / accuracy.
- Captura: el leaderboard del Model Lab + la curva de calibración.

**Slide 8 — La lectura senior**
- "Los modelos avanzados ganan por <1% de log-loss. Elo mantiene la mejor accuracy.
  Con ~64 partidos de señal, la complejidad no se paga."
- "`elo_diff` explica el 21,5% de la importancia en XGBoost — más que las 5
  siguientes features juntas."

**Slide 9 — Decisión**
- "Me quedo con el modelo interpretable — no por defecto, sino porque lo medí
  contra 3 alternativas legítimas y ganó donde importa (log-loss) prácticamente
  empatado, siendo el único explicable."

**Slide 10 — Cierre + links**
- Stack completo · link al dashboard · link al repo · "Construido por David Bravo".

---

## 3) Perfil permanente — que no se pierda en el feed

- **Sección "Destacado" (Featured):** perfil → botón "Añadir sección" → Destacado →
  Enlace. Pega la URL del dashboard. Aparece con la miniatura arriba de tu perfil.
  *Esto es lo más importante: es lo primero que ve un reclutador.*
- **Sección "Proyectos":** perfil → Añadir sección → Recomendado → Añadir proyecto.
  Título "FIFA World Cup Analytics — ML forecasting", asócialo a tu experiencia
  actual, pega ambos links y un resumen de 2 líneas.
- **Imagen social del repo (opcional):** GitHub → Settings → Social preview →
  sube una imagen 1280×640 (la captura del forecast sirve). Así el link se ve
  atractivo al compartirlo.

---

## Capturas recomendadas (tómalas tú a pantalla completa con Win+Shift+S)

Abre el dashboard en https://adavidbravo.github.io/fifa-worldcup-analytics/ (sin la
barra de WordPress) y captura, en este orden de impacto:
1. **Probabilidad de campeón** (barras con banderas) — la imagen #1 del post.
2. **Model Lab**: leaderboard de log-loss + curva de calibración — la prueba técnica.
3. **Bracket** "camino a la final" con porcentajes por llave.
4. **Carrera por la Bota de Oro** / títulos por país (banderas y colores).

---

## Timing

- **Feed post:** hoy o mañana, antes de semifinales (14-15 jul). El gancho
  "predije antes de que se jugara" pierde fuerza después.
- **Carrusel:** 2-3 días después del feed post, para no competir contigo mismo.
- Comenta tu propio post cuando el modelo acierte/falle cada llave — reactiva el
  alcance.
