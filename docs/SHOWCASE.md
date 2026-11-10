# Ideal foraging showcase

```bash
.venv/bin/python -m ant_homing showcase --output artifacts/foraging
```

Open `artifacts/foraging/replay.html` for play/pause, stepping and scrubbing, or `replay.gif` for the animation. Choose a new output directory for each run.

## Behaviour

The ant leaves its nest, follows a longer curved outbound route, reaches a food item, picks it up, and returns directly to the nest while carrying it. The food is deposited on delivery. Small movement increments and bounded turns replace the research baseline's repeated forward/reverse probes.

The scenario executes actual collision-checked simulator movements. Food location, carrying state, pickup, and delivery are recorded with the trajectory and synchronized with the camera and overhead renders. The return controller uses the noise-free accumulated displacement and compass reference to point home. Outbound exploration is scripted; discovering food autonomously is outside this demonstration.

## Relation to ant navigation

Many foraging ants can maintain a home vector and return fairly directly after finding food. Visual landmarks can guide or correct the return; uncertainty can produce searching near the nest. The baseline controller's repeated cardinal-direction sampling is a simple engineering algorithm, not a faithful model of typical ant locomotion.

This showcase depicts an idealized version of directed homing. It is not a validated model of ant nervous systems, walking biomechanics, or species-specific foraging. In particular, it does not show a trained neural network discovering how to forage.

## Separation from research

`showcase` is a separate command and records `demonstration_mode: ideal_foraging` in the summary. The replay labels its assumptions. Existing visual, neural, odometry and hybrid controllers remain available through `demo` and `benchmark`. Their failures and probing costs remain part of the recorded research results.
