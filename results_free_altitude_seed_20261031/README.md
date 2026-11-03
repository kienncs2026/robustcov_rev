# Free-altitude propulsion seed designs (31 Oct 2026)

These 13 runs (R12 propulsion-first designs and the two R2 propulsion-only rows) were produced with the
propulsion search *before* the level-flight projection of Remark 3 was added. The Powell search keeps
altitude oscillations: 1,017.0 J of vertical motion on the return-to-base mission and 174.0 J on the
reference mission. They are the "free-altitude seed" rows of Table III and the free-altitude-seed
paragraph of Section V-F (saving 17.47-17.55 % relative to hover, RF above hover <= 0.53 % of the
propulsion saving, violation bounds <= 0.131 %, 1,332.6 J above the level design; 5.92-J robustness
premium on this path in the response to R3.3).

The level-flight designs that replace them have the same configuration hash and live in
`results/R12/runs` and `results/R2/runs` (`propulsion_only_level = True` in `summary.csv`).
They were moved out of `results/` so that `python run_revision.py outputs` does not read both.
