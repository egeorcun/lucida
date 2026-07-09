import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_testset import _sanitize  # noqa: E402


def test_sanitize_url_unsafe_chars():
    assert _sanitize("a#b c(d).jpg-stem") == "a_b_c_d_.jpg-stem"


def test_sanitize_collapses_underscore_runs():
    assert _sanitize("x## ##y") == "x_y"


def test_sanitize_keeps_safe_chars():
    assert _sanitize("Ab-1_2.png") == "Ab-1_2.png"
