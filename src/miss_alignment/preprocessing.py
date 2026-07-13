"""Preprocessing utilities for tilt-series alignment."""

import json
import shutil
import torch
from pathlib import Path

from .data.io import TiltSeriesData
from .distributed.manager import run_distributed

# Sub-directory (under the training directory) where cross-correlation workers
# drop one JSON record per tilt-series when field-of-view pruning is enabled.
# ``run_cross_correlation_alignment_parallel`` aggregates these into the final
# ``preprocessing_dropped_views.json`` and then removes the directory. It lives
# under the training directory rather than the distributed queue root because
# ``run_distributed`` deletes the queue root on exit.
_DROPPED_VIEWS_DIRNAME = "preprocessing_dropped_views.d"


def _fov_fractions(
    shifts_px: torch.Tensor, image_shape: tuple[int, int]
) -> torch.Tensor:
    """Fraction of the reference frame each shifted view still overlaps.

    Cross-correlation shifts are axis-aligned in the image (YX) plane, so a view
    translated by ``(sy, sx)`` pixels overlaps its own un-shifted footprint over
    a rectangle of size ``(H - |sy|) x (W - |sx|)``. The returned fraction is that
    overlap area divided by the full image area, clamped to ``[0, 1]``.

    Parameters
    ----------
    shifts_px : torch.Tensor
        Per-view shifts in pixels, shape ``(n_tilts, 2)`` in YX order.
    image_shape : tuple[int, int]
        Image dimensions ``(H, W)`` in pixels.

    Returns
    -------
    torch.Tensor
        Field-of-view fraction per view in ``[0, 1]``, shape ``(n_tilts,)``.
    """
    h, w = image_shape
    dy = shifts_px[:, 0].abs().clamp(max=float(h))
    dx = shifts_px[:, 1].abs().clamp(max=float(w))
    return ((h - dy) / h) * ((w - dx) / w)


def _cross_correlate(
    stack: torch.Tensor,
    angles: torch.Tensor,
    tilt_axis_angle: float,
    pixel_size: float,
    lowpass_cutoff: float,
    device_str: str,
) -> torch.Tensor:
    """Run cross-correlation and return per-view shifts in pixels (YX)."""
    from torch_tiltxcorr import tiltxcorr

    return tiltxcorr(
        tilt_series=stack.to(device_str),
        tilt_angles=angles.to(device_str),
        tilt_axis_angle=tilt_axis_angle,
        pixel_spacing_angstroms=pixel_size,
        lowpass_angstroms=pixel_size / lowpass_cutoff,
    )


def _run_cross_correlation_single(
    xml_file: Path,
    device: int | None,
    lowpass_cutoff: float,
    prune_low_fov: bool = False,
    min_fov_fraction: float = 0.8,
    report_dir: Path | None = None,
) -> dict:
    """Run cross-correlation alignment on a single tilt-series.

    When ``prune_low_fov`` is set, the field-of-view fraction of each view is
    computed from the estimated shifts and stored in ``ts.fov_fraction``. Views
    whose fraction falls below ``min_fov_fraction`` are marked unused via
    ``ts.use_tilt`` (which the reconstruction path honours) and cross-correlation
    is re-run on the surviving subset so their shifts are not skewed by the
    dropped views. The image stack file itself is never modified.

    Parameters
    ----------
    xml_file : Path
        Path to the XML metadata file for the tilt-series.
    device : int | None
        CUDA device index to use. If None, uses default device.
    lowpass_cutoff : float
        Low-pass filter cutoff frequency.
    prune_low_fov : bool, optional
        Enable field-of-view based view pruning (default: False).
    min_fov_fraction : float, optional
        Minimum field-of-view fraction a view must retain to be kept
        (default: 0.8).
    report_dir : Path | None, optional
        If given, the per-series record is also written as
        ``<report_dir>/<xml_stem>.json`` so a parent process running this across
        many series (via the distributed worker pool) can aggregate the results.
        The record is always returned regardless of this argument.

    Returns
    -------
    dict
        Per-series record with keys ``xml_stem``, ``dropped_indices`` (original
        view indices marked unused) and ``n_tilts``.
    """
    if device is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)
        device_str = f"cuda:{device}"
    else:
        device_str = "cuda"

    # Load tilt-series data
    ts_data = TiltSeriesData(xml_metadata_path=xml_file)
    ts, stack, pixel_size = ts_data.load_metadata_and_stack(downsample=1)

    # Extract tilt axis angle from metadata (same for all tilts)
    tilt_axis_angle = ts.tilt_axis_angles[0].item()

    # Run cross-correlation alignment on the full stack
    shifts = _cross_correlate(
        stack, ts.angles, tilt_axis_angle, pixel_size, lowpass_cutoff, device_str
    )

    dropped_indices: list[int] = []
    if prune_low_fov:
        image_shape = (stack.shape[-2], stack.shape[-1])
        fov = _fov_fractions(shifts.detach().cpu(), image_shape)
        ts.fov_fraction = fov.to(torch.float32)
        keep = fov >= min_fov_fraction

        if not keep.any():
            # Every view is below threshold: disabling the whole series would
            # break reconstruction. Keep all views and flag it instead.
            print(
                f"WARNING: {xml_file.stem}: all {keep.numel()} views fall below "
                f"min_fov_fraction={min_fov_fraction}; keeping the full series "
                "unpruned."
            )
        elif not keep.all():
            ts.use_tilt = keep
            dropped_indices = torch.nonzero(~keep, as_tuple=True)[0].tolist()

            # Re-run cross-correlation on the surviving subset (in memory only;
            # tiltxcorr does not read use_tilt) and scatter the new shifts back
            # into the full-length array at the kept positions.
            keep_idx = torch.nonzero(keep, as_tuple=True)[0]
            sub_shifts = _cross_correlate(
                stack[keep_idx],
                ts.angles[keep_idx],
                tilt_axis_angle,
                pixel_size,
                lowpass_cutoff,
                device_str,
            )
            shifts = shifts.clone()
            shifts[keep_idx.to(shifts.device)] = sub_shifts.to(shifts.device)

            # Warn if the re-run still leaves survivors below threshold; we do
            # not iterate, but a silent partial pass would be misleading.
            residual = _fov_fractions(sub_shifts.detach().cpu(), image_shape)
            n_residual = int((residual < min_fov_fraction).sum())
            if n_residual:
                print(
                    f"WARNING: {xml_file.stem}: {n_residual} kept view(s) remain "
                    f"below min_fov_fraction={min_fov_fraction} after re-running "
                    "cross-correlation."
                )

    # Convert shifts from pixels to Angstroms
    shifts_angstrom = shifts * pixel_size

    # Apply shifts (note: negation and axis assignment)
    # shifts are in YX order, tilt_axis_offset are X and Y
    ts.tilt_axis_offset_x = -shifts_angstrom[:, 1]
    ts.tilt_axis_offset_y = -shifts_angstrom[:, 0]

    # Save updated metadata
    ts_data.save_metadata_to_xml(ts)

    record = {
        "xml_stem": xml_file.stem,
        "dropped_indices": dropped_indices,
        "n_tilts": int(len(ts.angles)),
    }

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{xml_file.stem}.json").write_text(json.dumps(record))

    return record


def run_cross_correlation_alignment_parallel(
    training_directory: Path,
    devices: list[int] | None = None,
    lowpass_cutoff: float = 0.25,
    n_cluster_workers: int | None = None,
    prune_low_fov: bool = False,
    min_fov_fraction: float = 0.8,
) -> None:
    """
    Run cross-correlation based alignment in parallel.

    This performs coarse alignment using cross-correlation to estimate shifts
    for all tilt-series in the training directory, processing multiple
    tilt-series in parallel across available GPUs.

    Parameters
    ----------
    training_directory : Path
        Directory containing XML metadata files for tilt-series.
    devices : list[int] | None, optional
        CUDA device indices to distribute work across (one worker process per
        unique device). If None, a single default-device worker is used.
    lowpass_cutoff : float, optional
        Low-pass filter cutoff frequency (default: 0.25).
    n_cluster_workers : int | None
        Number of cluster jobs to submit. When set, activates cluster mode;
        requires MISS_CLUSTER_CONFIG and MISS_CLUSTER_SCRIPT to be set.
    prune_low_fov : bool, optional
        Mark views whose field-of-view fraction falls below ``min_fov_fraction``
        as unused and re-run cross-correlation on the survivors (default: False).
    min_fov_fraction : float, optional
        Minimum field-of-view fraction a view must retain to be kept
        (default: 0.8). Only used when ``prune_low_fov`` is set.
    """
    # Get list of all XML files to process
    xml_files = list(training_directory.glob("*.xml"))

    if not xml_files:
        raise ValueError(f"No XML files found in {training_directory}")

    print(f"\nRunning cross-correlation alignment on {len(xml_files)} tilt-series...")
    print(f"  Low-pass cutoff: {lowpass_cutoff}")
    if prune_low_fov:
        print(f"  Pruning views below field-of-view fraction: {min_fov_fraction}")
    if devices:
        print(f"  Distributing across devices: {devices}\n")
    else:
        print("  Using default device assignment\n")

    # Workers drop one JSON record per series here when pruning is enabled; we
    # aggregate and delete it afterwards. Clear any stale directory up front.
    report_dir = training_directory / _DROPPED_VIEWS_DIRNAME
    if report_dir.exists():
        shutil.rmtree(report_dir)

    queue_root = training_directory / "tasks"
    run_distributed(
        tilt_series_list=xml_files,
        model_checkpoint=Path(""),
        output_directory=training_directory,
        setting="",
        patch_size=0,
        patch_overlap=0.0,
        batch_size=0,
        apply_ctf=False,
        downsample=1,
        devices=devices or [],
        n_cluster_workers=n_cluster_workers,
        queue_root=queue_root,
        task_type="cross_correlation",
        lowpass_cutoff=lowpass_cutoff,
        prune_low_fov=prune_low_fov,
        min_fov_fraction=min_fov_fraction,
    )

    if prune_low_fov:
        results: list[dict] = []
        if report_dir.exists():
            for record_path in sorted(report_dir.glob("*.json")):
                results.append(json.loads(record_path.read_text()))
            shutil.rmtree(report_dir)
        _report_pruned_views(training_directory, results)

    print("\nCross-correlation alignment complete!\n")


def _report_pruned_views(training_directory: Path, results: list[dict]) -> None:
    """Write a JSON summary of pruned views and print a short overview."""
    records = sorted(
        (r for r in results if isinstance(r, dict)),
        key=lambda r: r["xml_stem"],
    )
    total_dropped = sum(len(r["dropped_indices"]) for r in records)
    report_path = training_directory / "preprocessing_dropped_views.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "min_fov_fraction_pruning": True,
                "total_dropped": total_dropped,
                "series": records,
            },
            f,
            indent=2,
        )

    n_affected = sum(1 for r in records if r["dropped_indices"])
    print(
        f"  Pruned {total_dropped} view(s) across {n_affected} tilt-series; "
        f"summary written to {report_path}"
    )
