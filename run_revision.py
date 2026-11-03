"""Entry point: python run_revision.py <command> ...  (see README_REVISION_VI.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from uav_isac.revision_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
