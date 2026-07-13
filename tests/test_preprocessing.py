"""Tests for cross-correlation preprocessing and field-of-view pruning.

These do not require a GPU: ``_cross_correlate`` (the only part that touches
``torch_tiltxcorr``/CUDA) is monkeypatched to return deterministic shifts, so the
tests exercise the field-of-view geometry, the use-tilt masking, the re-run of
cross-correlation on survivors, and the summary report.
"""

from pathlib import Path

import mrcfile
import torch

from warpylib import TiltSeries

from miss_alignment import preprocessing
from miss_alignment.preprocessing import (
    _fov_fractions,
    _report_pruned_views,
    _run_cross_correlation_single,
)


def _make_tilt_series(tmp_path: Path, n_tilts: int, size: int = 100) -> Path:
    """Write a minimal TiltSeries XML + stack and return the XML path."""
    xml_path = tmp_path / "ts.xml"
    ts = TiltSeries(path=xml_path, n_tilts=n_tilts)
    ts.angles = torch.linspace(-60, 60, n_tilts)
    ts.tilt_axis_angles = torch.zeros(n_tilts)
    ts.image_dimensions_physical = torch.tensor([1000.0, 1000.0])
    ts.volume_dimensions_physical = torch.tensor([1000.0, 1000.0, 1000.0])
    ts.save_meta(xml_path)

    stack_path = Path(ts.tilt_stack_path)
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(stack_path, overwrite=True) as mrc:
        mrc.set_data(torch.randn(n_tilts, size, size).numpy())
        mrc.voxel_size = 1.0
    return xml_path


class TestFovFractions:
    def test_zero_shift_is_full_frame(self):
        fov = _fov_fractions(torch.zeros(3, 2), (100, 100))
        assert torch.allclose(fov, torch.ones(3))

    def test_axis_aligned_overlap(self):
        shifts = torch.tensor(
            [
                [0.0, 0.0],  # full
                [50.0, 0.0],  # half in Y -> 0.5
                [50.0, 50.0],  # half in both -> 0.25
                [-50.0, 0.0],  # sign does not matter -> 0.5
            ]
        )
        fov = _fov_fractions(shifts, (100, 100))
        assert torch.allclose(fov, torch.tensor([1.0, 0.5, 0.25, 0.5]))

    def test_shift_beyond_frame_clamps_to_zero(self):
        shifts = torch.tensor([[100.0, 0.0], [150.0, 0.0]])
        fov = _fov_fractions(shifts, (100, 100))
        assert torch.allclose(fov, torch.zeros(2))


class TestRunCrossCorrelationSingle:
    def test_prune_marks_views_reruns_and_leaves_stack_untouched(
        self, tmp_path, monkeypatch
    ):
        n_tilts = 6
        xml_path = _make_tilt_series(tmp_path, n_tilts, size=100)
        stack_bytes_before = Path(TiltSeries(xml_path).tilt_stack_path).read_bytes()

        calls = []

        def fake_cross_correlate(stack, angles, *_args, **_kwargs):
            calls.append(stack.shape[0])
            if stack.shape[0] == n_tilts:
                # Full pass: views 1 and 4 shift 90px in Y -> fov 0.1 (< 0.8).
                shifts = torch.zeros(n_tilts, 2)
                shifts[1, 0] = 90.0
                shifts[4, 0] = 90.0
                return shifts
            # Re-run on the 4 survivors: everything is well aligned.
            return torch.zeros(stack.shape[0], 2)

        monkeypatch.setattr(preprocessing, "_cross_correlate", fake_cross_correlate)

        record = _run_cross_correlation_single(
            xml_path,
            device=None,
            lowpass_cutoff=0.25,
            prune_low_fov=True,
            min_fov_fraction=0.8,
        )

        # Full pass then a re-run on the 4 survivors.
        assert calls == [n_tilts, 4]
        assert record["dropped_indices"] == [1, 4]
        assert record["n_tilts"] == n_tilts

        loaded = TiltSeries(xml_path)
        assert loaded.use_tilt.tolist() == [True, False, True, True, False, True]
        # fov_fraction is recorded from the full-pass shifts for every view.
        expected_fov = torch.tensor([1.0, 0.1, 1.0, 1.0, 0.1, 1.0])
        assert torch.allclose(loaded.fov_fraction, expected_fov, atol=1e-5)

        # The image stack file must never be modified.
        assert Path(loaded.tilt_stack_path).read_bytes() == stack_bytes_before

    def test_all_views_below_threshold_keeps_full_series(self, tmp_path, monkeypatch):
        n_tilts = 4
        xml_path = _make_tilt_series(tmp_path, n_tilts, size=100)

        def fake_cross_correlate(stack, *_args, **_kwargs):
            # Every view shifts 90px -> fov 0.1, all below threshold.
            shifts = torch.zeros(stack.shape[0], 2)
            shifts[:, 0] = 90.0
            return shifts

        monkeypatch.setattr(preprocessing, "_cross_correlate", fake_cross_correlate)

        record = _run_cross_correlation_single(
            xml_path,
            device=None,
            lowpass_cutoff=0.25,
            prune_low_fov=True,
            min_fov_fraction=0.8,
        )

        assert record["dropped_indices"] == []
        loaded = TiltSeries(xml_path)
        # Series is kept fully usable rather than disabling every view.
        assert loaded.use_tilt.all()

    def test_pruning_disabled_leaves_use_tilt_default(self, tmp_path, monkeypatch):
        n_tilts = 5
        xml_path = _make_tilt_series(tmp_path, n_tilts, size=100)

        def fake_cross_correlate(stack, *_args, **_kwargs):
            shifts = torch.zeros(stack.shape[0], 2)
            shifts[0, 0] = 95.0  # would be pruned if pruning were enabled
            return shifts

        monkeypatch.setattr(preprocessing, "_cross_correlate", fake_cross_correlate)

        record = _run_cross_correlation_single(
            xml_path,
            device=None,
            lowpass_cutoff=0.25,
            prune_low_fov=False,
        )

        assert record["dropped_indices"] == []
        loaded = TiltSeries(xml_path)
        assert loaded.use_tilt.all()


def test_report_pruned_views_writes_summary(tmp_path):
    results = [
        {"xml_stem": "b", "dropped_indices": [2], "n_tilts": 10},
        {"xml_stem": "a", "dropped_indices": [], "n_tilts": 10},
        {"xml_stem": "c", "dropped_indices": [0, 7], "n_tilts": 10},
    ]
    _report_pruned_views(tmp_path, results)

    report_path = tmp_path / "preprocessing_dropped_views.json"
    assert report_path.exists()

    import json

    data = json.loads(report_path.read_text())
    assert data["total_dropped"] == 3
    # Series are sorted by name for stable output.
    assert [s["xml_stem"] for s in data["series"]] == ["a", "b", "c"]
