"""Experiment plans build without a solver and have unique run identities."""
from pathlib import Path

from uav_isac.config import load_config
from uav_isac.experiments import EXPERIMENTS, build
from uav_isac.io_utils import config_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_all_plans_unique():
    base = load_config(ROOT / "configs" / "reference_strict.json")
    seen = set()
    for exp_id in EXPERIMENTS:
        for run in build(exp_id, base, {"n_local": 3, "n_unrestricted": 2}):
            fp = config_fingerprint(run.config, length=0)
            assert fp not in seen, run.label
            seen.add(fp)


def test_legacy_reference_fingerprint_unchanged():
    """Default revision fields must not change the identity of a locked run."""
    from uav_isac.legacy.postlock import build_plan
    base = load_config(ROOT / "configs" / "reference_strict.json")
    t0 = [p for p in build_plan(base, ["S9"]) if p.topology_seed == 0 and p.method_id == "proposed"][0]
    assert config_fingerprint(t0.config) == "2137019914ce"  # archived run directory of Table II
