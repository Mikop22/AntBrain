# Ideal foraging showcase

```bash
.venv/bin/python -m ant_homing showcase --food-seed 0 --search-seed 0 --output artifacts/foraging
```

Generates `replay.html` (interactive player) and `replay.gif` for an idealized foraging scenario:

- **Food-blind pillar search (`search.py`):** Food is placed in a pocket behind a pillar occluded from the nest. The policy visits unvisited inspection sites around pillars without prior knowledge of the food coordinates.
- **Line-of-sight detection (`FoodSensor`):** A world-side geometric sensor reveals the food position only when the ant is within $0.60\text{ m}$ with an unobstructed line of sight.
- **Obstacle-aware return (`showcase.py`):** After pickup, a visibility-graph planner routes the ant and carried cargo around pillars back to the nest.

This command (`demonstration_mode: ideal_foraging`) is an engine showcase using known obstacle geometry and exact self-position, separate from the sensor-restricted research controllers in `navigation.py`.
