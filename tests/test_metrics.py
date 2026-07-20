import numpy as np
import pytest

from benchmark.metrics import all_metrics, bg_stats, conn_error, grad_error, mae, mse, sad


@pytest.fixture
def square_alpha():
    gt = np.zeros((100, 100), dtype=np.float32)
    gt[25:75, 25:75] = 1.0
    return gt


def test_identical_alphas_give_zero(square_alpha):
    for fn in (sad, mae, mse, grad_error, conn_error):
        assert fn(square_alpha, square_alpha) == pytest.approx(0.0, abs=1e-6)


def test_sad_counts_absolute_difference(square_alpha):
    pred = square_alpha.copy()
    pred[0, 0:10] = 0.5  # 10 pixels, 0.5 difference -> SAD = 5/1000
    assert sad(pred, square_alpha) == pytest.approx(0.005)


def test_mae_and_mse(square_alpha):
    pred = np.clip(square_alpha + 0.1, 0, 1).astype(np.float32)
    assert mae(pred, square_alpha) == pytest.approx(0.075, abs=0.01)
    assert mse(pred, square_alpha) < mae(pred, square_alpha)


def test_grad_penalizes_blurry_edges(square_alpha):
    from scipy import ndimage
    blurry = ndimage.gaussian_filter(square_alpha, sigma=3).astype(np.float32)
    shifted = np.roll(square_alpha, 1, axis=0)
    assert grad_error(blurry, square_alpha) > 0
    assert grad_error(shifted, square_alpha) > 0


def test_conn_penalizes_disconnected_blobs(square_alpha):
    disconnected = square_alpha.copy()
    disconnected[5:10, 5:10] = 1.0  # small blob disconnected from the main square
    assert conn_error(disconnected, square_alpha) > 0


def test_all_metrics_keys(square_alpha):
    m = all_metrics(square_alpha, square_alpha)
    assert set(m) == {"sad", "mae", "mse", "grad", "conn", "bg_mae", "bg_smear"}


def test_bg_stats_clean_background_is_zero(square_alpha):
    m = bg_stats(square_alpha, square_alpha)
    assert m["bg_mae"] == pytest.approx(0.0)
    assert m["bg_smear"] == pytest.approx(0.0)


def test_bg_stats_measures_haze_missed_by_mae(square_alpha):
    """A faint 0.1 haze over the whole background: overall MAE stays small but
    bg_mae/bg_smear must flag it — this is the metric's reason to exist."""
    pred = square_alpha.copy()
    pred[square_alpha == 0.0] = 0.1
    m = bg_stats(pred, square_alpha)
    assert m["bg_mae"] == pytest.approx(0.1, abs=1e-6)
    assert m["bg_smear"] == pytest.approx(1.0)


def test_bg_stats_erosion_excludes_soft_edge(square_alpha):
    """Residue only in the edge band (within erosion_px of the subject) must
    NOT count: the eroded region measures unambiguous background only."""
    pred = square_alpha.copy()
    pred[20:25, 25:75] = 0.5  # 5px strip hugging the square's top edge
    m = bg_stats(pred, square_alpha)
    assert m["bg_mae"] == pytest.approx(0.0)


def test_bg_stats_nan_when_no_background():
    gt = np.ones((100, 100), dtype=np.float32)  # subject fills the frame
    m = bg_stats(gt, gt)
    assert np.isnan(m["bg_mae"]) and np.isnan(m["bg_smear"])


def test_all_metrics_omits_bg_keys_when_unmeasurable():
    """No NaN may reach the per-image dicts (metrics.json must stay valid
    strict JSON and dict-equality must round-trip): when there is no
    measurable background, the bg keys are simply absent."""
    gt = np.ones((100, 100), dtype=np.float32)
    m = all_metrics(gt, gt)
    assert set(m) == {"sad", "mae", "mse", "grad", "conn"}
