"""Preprocessing utilities for tilt-series alignment."""

import json
import queue
import torch
from pathlib import Path

from ._parallel import run_device_pool
from .data.io import TiltSeriesData


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

    return {
        "xml_stem": xml_file.stem,
        "dropped_indices": dropped_indices,
        "n_tilts": int(len(ts.angles)),
    }


def _cross_correlation_runner(
    device: int | None,
    task_queue,
    result_queue,
    lowpass_cutoff: float,
    prune_low_fov: bool,
    min_fov_fraction: float,
) -> None:
    """Pull tilt-series off the queue and align them on a single device."""

    torch.set_num_threads(1)
    while True:
        try:
            xml_file = task_queue.get_nowait()
        except queue.Empty:
            break
        record = _run_cross_correlation_single(
            xml_file,
            device,
            lowpass_cutoff,
            prune_low_fov,
            min_fov_fraction,
        )
        result_queue.put_nowait(record)


def run_cross_correlation_alignment_parallel(
    training_directory: Path,
    devices: list[int] | None = None,
    lowpass_cutoff: float = 0.25,
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

    results = run_device_pool(
        jobs=xml_files,
        runner=_cross_correlation_runner,
        runner_args=(lowpass_cutoff, prune_low_fov, min_fov_fraction),
        devices=devices,
        desc="Cross-correlation alignment",
    )

    if prune_low_fov:
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
