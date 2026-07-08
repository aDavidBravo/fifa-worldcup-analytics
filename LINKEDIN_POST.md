# LinkedIn post draft

---

⚽ **Who wins the 2026 World Cup? I built a model to find out.**

While the quarter-finals kick off this week, I shipped an end-to-end data science
project covering **96 years of World Cup history — and a live forecast of the
tournament being played right now**.

🔮 My Elo + Monte Carlo model (10,000 simulations) says:
→ 🇦🇷 Argentina 28.9%
→ 🇪🇸 Spain 27.4%
→ 🇫🇷 France 19.7%
→ 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England 12.6%

What's under the hood:
📊 1,060 matches and 3,000 goals analyzed, from Uruguay 1930 to the round of 16 in 2026
🧮 Elo ratings computed over 49,000+ international matches since 1872, weighted by
match importance and goal margin
🎲 The confirmed bracket simulated 10,000 times with logistic win expectancy
📈 An interactive dark-mode dashboard: championship odds, bracket probabilities,
the Messi–Mbappé–Haaland Golden Boot race, confederation dominance (UEFA 12 titles
vs CONMEBOL 10), and the all-time records

Stack: **Python (pandas/NumPy) → JSON pipeline → Apache ECharts**. Fully
reproducible, zero backend, one command to refresh as results come in.

🔗 Live dashboard: [YOUR-GITHUB-PAGES-URL]
💻 Code: [YOUR-REPO-URL]

Will the model beat the pundits? We'll know on July 19 at MetLife Stadium. 🏆

#DataScience #Python #Analytics #WorldCup2026 #DataVisualization #MachineLearning #SportsAnalytics

---

**Tips:**
- Publica antes de las semifinales (14-15 julio) para máxima relevancia; actualiza
  el post con un comentario cuando el modelo acierte/falle cada llave.
- Adjunta 2-3 capturas del dashboard (la de probabilidades de campeón primero).
- Reemplaza los placeholders de URL cuando subas el repo a GitHub y actives Pages
  (Settings → Pages → branch main, folder /dashboard... o mueve dashboard/ a docs/).
