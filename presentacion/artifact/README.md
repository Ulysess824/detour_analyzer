Source of the "Planificador contra modelo" deck (Slides artifact on claude.ai, 8 slides, white background, Smurfit Westrock logo and colors).

- `project/deck.json`: slide order and fonts.
- `project/slides/<id>.html`: one slide each (1920x1080 canvas, inline styles). The logo is referenced as an uploaded asset (`/_blob/...`); the same image is `../logo_smurfit_westrock.png`.
- Numbers are typed by hand from `results/planner_comparison_*.csv` (rolling-origin test, `ml_mean`). The LaTeX version in `../presentacion.tex` is older (no method, metric and variables slides).
