# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance.
# Checks: Published profiles select the documented retrieval paths.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""CPU-only checks that published profiles select the adopted retrieval paths."""
from pathlib import Path

import pytest

from leap.common.config import RetrievalConfig


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("hausdorff", RetrievalConfig("hausdorff")),
        ("dfd", RetrievalConfig("dfd", dfd_initial="light")),
        (
            "dtw",
            RetrievalConfig(
                "dtw", dtw_seed="signal", dtw_pool=128,
                dtw_routing="learned", dtw_execution="optimized",
            ),
        ),
    ],
)
def test_adopted_profile(name, expected):
    assert RetrievalConfig.from_json(ROOT / "02_configs" / name / f"{name}.json") == expected


def test_legacy_dtw_default_remains_explicitly_distinct():
    legacy = RetrievalConfig("dtw")
    adopted = RetrievalConfig.from_json(ROOT / "02_configs/dtw/dtw.json")
    assert (legacy.dtw_seed, legacy.dtw_routing) == ("weighted", "fixed")
    assert adopted != legacy


def test_readme_links_all_adopted_profiles():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "(11_docs/configuration.md)" in readme
    guide = (ROOT / "11_docs/configuration.md").read_text(encoding="utf-8")
    assert "(../02_configs/README.md)" in guide
    profiles = (ROOT / "02_configs/README.md").read_text(encoding="utf-8")
    for name in ("hausdorff", "dfd", "dtw"):
        assert f"({name}/{name}.json)" in profiles
        assert (ROOT / "02_configs" / name / f"{name}.json").is_file()
