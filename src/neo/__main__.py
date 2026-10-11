"""Allow running Neo as a module: python -m neo"""

import sys

from neo.cli import main

if __name__ == "__main__":
    # main() returns the exit status; dropping it made `python -m neo`
    # report success for every refusal and failure.
    sys.exit(main())
