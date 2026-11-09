"""Allow ``python -m ant_homing`` as an alternative to the console script."""

from .cli import main

raise SystemExit(main())
