"""Allow the same commands through python -m scal."""

from .cli import main

raise SystemExit(main())
