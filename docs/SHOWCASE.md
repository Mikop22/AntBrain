# Ideal foraging showcase

```bash
.venv/bin/python -m ant_homing showcase --output artifacts/foraging
```

Open `artifacts/foraging/replay.html` for play/pause, stepping and scrubbing, or `replay.gif` for the animation. Choose a new output directory for each run.

Use the default 6 m arena or a larger one for this scenario. Tight custom scenes can lack an accessible pocket behind a pillar once ant and cargo clearance are included; these configurations report an error rather than place the food inside an obstacle.

Each CLI invocation chooses a fresh food-placement seed and records it in the summary. For a repeatable episode, supply `--food-seed 0 --search-seed 0`. The scene seed (`--seed`), food seed, and search tie-breaking seed are separate. Change only the food seed to test whether the search remains independent of the hidden target. `--max-steps` bounds the entire trip; timeout and exhausted search are failures, not successful deliveries.

## Behaviour

The ant leaves its nest without knowing where food was placed. It visits inspection sites around the nearest unvisited pillar, remembers completed sites, and continues to other pillars when a search is empty. It stops searching only after a local detection, approaches the observed food, picks it up, and returns home around obstacles. The food is deposited on delivery. Small forward movements and bounded turns replace the research baseline's repeated forward/reverse probes.

Food placement is randomly selected from reachable pockets behind actual pillars. A line from the nest to the food crosses its pillar. The search policy receives the same catalogue of inspection sites for every food placement, with no selected food location or pillar. It chooses nearby unvisited sites with seeded tie-breaking and uses visited-site memory rather than repeating completed checks.

A separate world-side sensor reveals the food coordinates only within its fixed detection range and with an unobstructed sightline past every pillar. This is ideal geometric perception, not learned object recognition or an odour model. Before that detection, the food location cannot guide search actions. The replay's food highlight also stays hidden until detection, although the physical object remains in the rendered scene.

Routes use ideal scene-geometry planning with additional clearance for the carried object. The scenario executes collision-checked movements; the ant never passes through a pillar to return home. The showcase still assumes a known obstacle layout, exact self-position, and ideal actuation. The improvement removes advance knowledge of the food; it does not establish biologically accurate exploration or learned navigation.

Food location, first detection, inspection memory, carrying state, pickup, and delivery are recorded with the trajectory and synchronized with the camera and overhead renders. The summary records the chosen pillar, seeds, sensor conditions, and route-planning assumptions. An unsuitable scene can report a clear error instead of placing inaccessible food.

## Relation to ant navigation

Many foraging ants can maintain a home vector and return fairly directly after finding food. Visual landmarks can guide or correct the return; uncertainty can produce searching near the nest. The baseline controller's repeated cardinal-direction sampling is a simple engineering algorithm, not a faithful model of typical ant locomotion.

This showcase depicts an idealized version of directed homing. It is not a validated model of ant nervous systems, walking biomechanics, or species-specific foraging. In particular, it does not show a trained neural network discovering how to forage.

## Separation from research

`showcase` is a separate command and records `demonstration_mode: ideal_foraging` in the summary. The GIF has a compact layout with only arena, camera, and route labels; assumptions remain in the HTML description, summary, and documentation. Existing visual, neural, odometry and hybrid controllers remain available through `demo` and `benchmark`. Their failures and probing costs remain part of the recorded research results.
