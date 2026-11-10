# Ideal foraging showcase

```bash
.venv/bin/python -m ant_homing showcase --output artifacts/foraging
```

Open `artifacts/foraging/replay.html` for play/pause, stepping and scrubbing, or `replay.gif` for the animation. Choose a new output directory for each run.

Use the default 6 m arena or a larger one for this scenario. Tight custom scenes can lack an accessible pocket behind a pillar once ant and cargo clearance are included; these configurations report an error rather than place the food inside an obstacle.

## Behaviour

The ant leaves its nest, winds around landmarks, reaches food on the far side of a pillar, picks it up, and returns around the obstruction while carrying it. The food is deposited on delivery. Small forward movements and bounded turns replace the research baseline's repeated forward/reverse probes.

The food placement is chosen from actual pillar geometry: a line from the nest to the food crosses the pillar itself, and the pickup location remains reachable. Both routes use obstacle-aware planning with additional clearance for the carried object. The scenario executes actual collision-checked movements; the ant never passes through the pillar to return home. Outbound exploration and access to scene geometry are explicit engine-showcase assumptions, not learned food discovery or biological sensing.

Food location, carrying state, pickup, and delivery are recorded with the trajectory and synchronized with the camera and overhead renders. The summary records the chosen pillar and route-planning assumptions. An unsuitable scene can report a clear error instead of placing inaccessible food.

## Relation to ant navigation

Many foraging ants can maintain a home vector and return fairly directly after finding food. Visual landmarks can guide or correct the return; uncertainty can produce searching near the nest. The baseline controller's repeated cardinal-direction sampling is a simple engineering algorithm, not a faithful model of typical ant locomotion.

This showcase depicts an idealized version of directed homing. It is not a validated model of ant nervous systems, walking biomechanics, or species-specific foraging. In particular, it does not show a trained neural network discovering how to forage.

## Separation from research

`showcase` is a separate command and records `demonstration_mode: ideal_foraging` in the summary. The GIF has a compact layout with only arena, camera, and route labels; assumptions remain in the HTML description, summary, and documentation. Existing visual, neural, odometry and hybrid controllers remain available through `demo` and `benchmark`. Their failures and probing costs remain part of the recorded research results.
