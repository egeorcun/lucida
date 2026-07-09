import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_testset import _sanitize, classify_disvd, parse_disvd_class  # noqa: E402


def test_sanitize_url_unsafe_chars():
    assert _sanitize("a#b c(d).jpg-stem") == "a_b_c_d_.jpg-stem"


def test_sanitize_collapses_underscore_runs():
    assert _sanitize("x## ##y") == "x_y"


def test_sanitize_keeps_safe_chars():
    assert _sanitize("Ab-1_2.png") == "Ab-1_2.png"


def test_parse_disvd_class_from_sanitized_stem():
    assert parse_disvd_class("11_Furniture_7_Desk_5888267534_272111a3ac_o") == "Desk"


def test_parse_disvd_class_from_full_id_with_multiword_group():
    stem = "disvd_complex_17_Non-motor_Vehicle_1_BabyCarriage_28164603627_fd620eb85c_o"
    assert parse_disvd_class(stem) == "BabyCarriage"


def test_classify_disvd_thin_and_complex():
    assert classify_disvd("disvd_thin_20_Sports_8_Racket_4827171149_3140bffe12_o") == "thin"
    assert classify_disvd("disvd_complex_11_Furniture_4_Chair_6058764304_34287f447b_o") == "complex"


def test_classify_disvd_unparseable_defaults_complex():
    assert classify_disvd("not_a_disvd_stem") == "complex"
