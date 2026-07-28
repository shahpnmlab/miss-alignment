"""Calibrate a trained MissAlignment score against physical misalignment.

The alignment score is trained with a purely *contrastive* objective: the loss
only constrains the ordering of aligned vs misaligned patches and the gap
between them, never the absolute value. The scale is therefore arbitrary, drifts
between macro-iterations, and is confounded by specimen content (thickness,
contrast, ice). That makes a raw score threshold for including/excluding
tilt-series ill-posed.

This script recovers a physical scale by *injecting a known misalignment* and
measuring how the score responds, one tilt-series at a time.

The model
---------
Injecting a random per-tilt shift of size ``delta`` on top of a tilt-series that
still carries an unknown residual error ``r`` gives, for independent errors, a
total error that adds in quadrature::

    total(delta) = sqrt(r**2 + delta**2)

Assuming the score responds approximately linearly to the total error over the
sampled range, the measured curve is::

    score(delta) = alpha + beta * sqrt(r**2 + delta**2)

``alpha`` and ``beta`` absorb the arbitrary score offset and scale (and with
them most of the specimen-content dependence), leaving ``r`` -- an estimate of
the residual alignment error **in Angstroms**, which is comparable across
tilt-series.

The intuition is worth stating plainly: a series that is *already* badly
misaligned barely notices a small extra shift, so its curve is **flat** near
delta=0. A well-aligned series degrades immediately, so its curve is **steep**.
The flatness near zero is what encodes ``r``, and it is a shape in Angstroms,
not a position on the arbitrary score axis.

Caveats
-------
- ``r`` is only constrained when the sampled ``delta`` values are comparable to
  it. If every sampled delta is far larger than ``r`` the curve looks linear and
  ``r`` collapses toward zero with a large uncertainty. Check ``r_stderr`` and
  the reported fit quality before trusting a value.
- The quadrature and local-linearity assumptions are approximations. Treat the
  output as a calibrated ranking with physical units, not a metrology result.
- Scores are only comparable within one model checkpoint. Always calibrate with
  the checkpoint whose scores you intend to threshold.

Example
-------
    python examples/calibrate_alignment_score.py \\
        --model-checkpoint run/iter3/model.ckpt \\
        --tilt-series 'tiltseries/*.xml' \\
        --output-directory calibration/ \\
        --device cuda:0
"""

import json
import math
from pathlib import Path
from typing import Optional

import einops
import numpy as np
import torch
import typer
from scipy.optimize import curve_fit
from warpylib.tilt_series.reconstruct_volume import preprocess_tilt_data

from miss_alignment.alignment.tilt_series import generate_position_grid
from miss_alignment.data.io import TiltSeriesData
from miss_alignment.models import MissAlignment

# Sub-pixel to few-pixel spacing, in units of the (downsampled) pixel size.
# The small values are what constrain the residual error; the large ones pin
# down the slope.
DEFAULT_DELTA_STEPS_PIXELS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def load_model(model_checkpoint: Path, device: str) -> MissAlignment:
    """Load a trained checkpoint in eager mode, ready for scoring."""
    model = MissAlignment.load_from_checkpoint(model_checkpoint, map_location="cpu")
    # load_from_checkpoint calls configure_model(), which wraps self.net in
    # torch.compile. Unwrap for eager-mode inference, as evaluate_tilt_series does.
    if hasattr(model.net, "_orig_mod"):
        model.net = model.net._orig_mod
    model.to(device)
    model.freeze()
    model.eval()
    return model


@torch.no_grad()
def score_tilt_series(
    model: MissAlignment,
    tilt_series,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    patch_size: int,
    batch_size: int,
    apply_ctf: bool,
    device: str,
) -> float:
    """Precision-weighted mean score over a set of patch positions.

    This reproduces the quantity that the alignment stage reports as the
    per-tilt-series "alignment loss" (see ``optimize_global.optimize_shifts``),
    without any optimization: reconstruct patches, normalize each one, run the
    model, and average the scores weighted by the predicted precision.
    """
    n_batches = int(math.ceil(positions.shape[0] / batch_size))
    total_weighted_score = 0.0
    total_precision = 0.0

    with torch.amp.autocast(
        device_type="cuda" if str(device).startswith("cuda") else "cpu", enabled=False
    ):
        for b in range(n_batches):
            batch_positions = positions[b * batch_size : (b + 1) * batch_size]

            subvolumes = tilt_series.reconstruct_subvolumes_single(
                tilt_data=images,
                coords=batch_positions.to(device),
                pixel_size=pixel_size,
                size=patch_size,
                apply_ctf=apply_ctf,
                oversampling=2.0,
            )

            # per-subvolume normalization, matching the alignment stage
            mean = einops.reduce(subvolumes, "n d h w -> n 1 1 1", reduction="mean")
            std = torch.std(subvolumes, dim=(-3, -2, -1), keepdim=True)
            subvolumes = (subvolumes - mean) / std.clamp(min=1e-6)
            subvolumes = einops.rearrange(subvolumes, "b d h w -> b 1 d h w")

            scores, log_precisions = model(subvolumes)
            precisions = log_precisions.exp()

            total_weighted_score += (scores * precisions).sum().item()
            total_precision += precisions.sum().item()

    if total_precision <= 0:
        raise ValueError(
            f"Total precision is {total_precision}, which is <= 0; the model "
            "precision outputs are degenerate for this tilt-series."
        )
    return total_weighted_score / total_precision


def inject_misalignment(
    tilt_series,
    baseline_offset_x: torch.Tensor,
    baseline_offset_y: torch.Tensor,
    sigma_angstrom: float,
    generator: torch.Generator,
) -> None:
    """Add independent Gaussian per-tilt shifts on top of the baseline offsets.

    The perturbation is applied in the image plane, which is where
    ``tilt_axis_offset_x/y`` live and where a residual alignment error physically
    manifests. Only tilts that are enabled via ``use_tilt`` are perturbed:
    disabled tilts are excluded from reconstruction, so shifting them would both
    be a no-op and corrupt the accounting of the injected magnitude.
    """
    n_tilts = baseline_offset_x.shape[0]
    use_tilt = tilt_series.use_tilt.detach().cpu()

    noise_x = torch.normal(
        mean=0.0, std=sigma_angstrom, size=(n_tilts,), generator=generator
    )
    noise_y = torch.normal(
        mean=0.0, std=sigma_angstrom, size=(n_tilts,), generator=generator
    )
    noise_x[~use_tilt] = 0.0
    noise_y[~use_tilt] = 0.0

    device = baseline_offset_x.device
    tilt_series.tilt_axis_offset_x = baseline_offset_x + noise_x.to(device)
    tilt_series.tilt_axis_offset_y = baseline_offset_y + noise_y.to(device)


def _quadrature_response(delta, alpha, beta, r):
    """score(delta) = alpha + beta * sqrt(r^2 + delta^2)."""
    return alpha + beta * np.sqrt(r**2 + delta**2)


def fit_residual_error(
    deltas: np.ndarray, scores: np.ndarray
) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """Fit the quadrature model and return (r, r_stderr, r_squared, note)."""
    span = float(scores.max() - scores.min())
    if span <= 0:
        return None, None, None, "score did not vary with injected shift"

    # beta > 0: a larger misalignment must give a larger (less negative) score.
    p0 = [
        float(scores.min()),
        span / max(float(deltas.max()), 1e-6),
        float(deltas.max()) / 2,
    ]
    try:
        popt, pcov = curve_fit(
            _quadrature_response,
            deltas,
            scores,
            p0=p0,
            bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError) as exc:
        return None, None, None, f"fit failed: {exc}"

    residuals = scores - _quadrature_response(deltas, *popt)
    ss_res = float((residuals**2).sum())
    ss_tot = float(((scores - scores.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    r = float(popt[2])
    r_stderr = float(np.sqrt(np.diag(pcov))[2]) if np.all(np.isfinite(pcov)) else None

    note = "ok"
    if r_stderr is not None and r_stderr > r:
        note = "poorly constrained: sample smaller injected shifts"
    elif r > 0.8 * float(deltas.max()):
        note = "at edge of sampled range: sample larger injected shifts"
    return r, r_stderr, r_squared, note


def calibrate_one(
    model: MissAlignment,
    xml_path: Path,
    patch_size: int,
    patch_overlap: float,
    batch_size: int,
    apply_ctf: bool,
    downsample: int,
    device: str,
    max_positions: int,
    n_repeats: int,
    delta_steps_pixels: tuple[float, ...],
    seed: int,
) -> dict:
    """Measure the score-vs-injected-misalignment curve for one tilt-series."""
    ts_data = TiltSeriesData(xml_metadata_path=xml_path)
    tilt_series, images, pixel_size = ts_data.load_metadata_and_stack(
        downsample=downsample
    )
    images = preprocess_tilt_data(
        tilt_data=images, normalize=True, invert=False, subvolume_size=patch_size
    )
    tilt_series.to(device)
    images = images.to(device)

    positions = generate_position_grid(
        volume_dimensions_physical=tilt_series.volume_dimensions_physical,
        pixel_size=pixel_size,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )

    # Use one fixed subset of positions for every measurement. The paired design
    # removes position-to-position variance from the curve, which matters far
    # more here than the absolute number of patches.
    generator = torch.Generator().manual_seed(seed)
    if positions.shape[0] > max_positions:
        idx = torch.randperm(positions.shape[0], generator=generator)[:max_positions]
        positions = positions[idx]

    baseline_offset_x = tilt_series.tilt_axis_offset_x.clone()
    baseline_offset_y = tilt_series.tilt_axis_offset_y.clone()

    deltas_angstrom = [d * pixel_size for d in delta_steps_pixels]
    curve = []
    for delta in deltas_angstrom:
        # delta=0 is deterministic, so one evaluation is enough
        repeats = 1 if delta == 0.0 else n_repeats
        samples = []
        for _ in range(repeats):
            inject_misalignment(
                tilt_series,
                baseline_offset_x,
                baseline_offset_y,
                sigma_angstrom=delta,
                generator=generator,
            )
            samples.append(
                score_tilt_series(
                    model=model,
                    tilt_series=tilt_series,
                    images=images,
                    pixel_size=pixel_size,
                    positions=positions,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    apply_ctf=apply_ctf,
                    device=device,
                )
            )
        curve.append(
            {
                "delta_angstrom": float(delta),
                "delta_pixels": float(delta / pixel_size),
                "score_mean": float(np.mean(samples)),
                "score_std": float(np.std(samples)) if len(samples) > 1 else 0.0,
                "n_samples": len(samples),
            }
        )

    # restore the baseline so the in-memory object is never left perturbed
    tilt_series.tilt_axis_offset_x = baseline_offset_x
    tilt_series.tilt_axis_offset_y = baseline_offset_y

    deltas = np.array([p["delta_angstrom"] for p in curve])
    scores = np.array([p["score_mean"] for p in curve])
    r, r_stderr, r_squared, note = fit_residual_error(deltas, scores)

    # Local sensitivity: score units per Angstrom over the first two points.
    # This is what converts a score gap into an equivalent shift.
    if len(curve) > 1 and deltas[1] > 0:
        sensitivity = float((scores[1] - scores[0]) / (deltas[1] - deltas[0]))
    else:
        sensitivity = float("nan")

    n_used = int(tilt_series.use_tilt.sum())
    return {
        "tilt_series": xml_path.stem,
        "pixel_size_angstrom": float(pixel_size),
        "n_tilts_total": int(tilt_series.use_tilt.numel()),
        "n_tilts_used": n_used,
        "n_positions": int(positions.shape[0]),
        "score_at_zero": float(scores[0]),
        "residual_error_angstrom": r,
        "residual_error_stderr": r_stderr,
        "fit_r_squared": r_squared,
        "fit_note": note,
        "sensitivity_score_per_angstrom": sensitivity,
        "curve": curve,
    }


def plot_curves(results: list[dict], output_path: Path) -> None:
    """Plot the score-vs-injected-misalignment curves, normalized for overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(13, 5))

    for result in results:
        deltas = np.array([p["delta_angstrom"] for p in result["curve"]])
        scores = np.array([p["score_mean"] for p in result["curve"]])
        errs = np.array([p["score_std"] for p in result["curve"]])
        label = result["tilt_series"]

        ax_raw.errorbar(deltas, scores, yerr=errs, marker="o", capsize=3, label=label)

        # Normalize away the arbitrary offset and scale so the *shape* -- which
        # is what carries the physical information -- can be compared directly.
        span = scores.max() - scores.min()
        if span > 0:
            ax_norm.plot(
                deltas, (scores - scores.min()) / span, marker="o", label=label
            )

    ax_raw.set_xlabel("injected misalignment (Å, per-axis std)")
    ax_raw.set_ylabel("precision-weighted score")
    ax_raw.set_title("Raw score response")
    ax_raw.grid(alpha=0.3)

    ax_norm.set_xlabel("injected misalignment (Å, per-axis std)")
    ax_norm.set_ylabel("normalized score response")
    ax_norm.set_title("Shape comparison (flat near 0 = already misaligned)")
    ax_norm.grid(alpha=0.3)

    if len(results) <= 12:
        ax_raw.legend(fontsize=8)
        ax_norm.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(
    model_checkpoint: Path = typer.Option(
        ..., help="Path to a trained model checkpoint (e.g. run/iter3/model.ckpt)."
    ),
    tilt_series: str = typer.Option(
        ...,
        help="Glob for tilt-series XML files, e.g. 'tiltseries/*.xml'. "
        "Quote it so the shell does not expand it.",
    ),
    output_directory: Path = typer.Option(
        Path("calibration"), help="Directory for the JSON report and plot."
    ),
    patch_size: int = typer.Option(96, help="Match tilt_series_alignment.patch_size."),
    patch_overlap: float = typer.Option(0.1, help="Match the alignment config."),
    batch_size: int = typer.Option(32, help="Patches reconstructed simultaneously."),
    apply_ctf: bool = typer.Option(False, help="Match general.apply_ctf."),
    downsample: int = typer.Option(
        1, help="Match the downsample of the iteration being calibrated."
    ),
    device: str = typer.Option("cuda:0", help="Device to run on."),
    max_positions: int = typer.Option(
        64,
        help="Patch positions per measurement. Cost scales linearly with this; "
        "the paired design keeps the curve smooth even at modest values.",
    ),
    n_repeats: int = typer.Option(
        5, help="Random realizations averaged per non-zero injected shift."
    ),
    max_delta_pixels: Optional[float] = typer.Option(
        None,
        help="Override the largest injected shift, in pixels of the "
        "(downsampled) stack. Default sweeps up to 3 pixels.",
    ),
    seed: int = typer.Option(42, help="Seed for positions and injected noise."),
) -> None:
    """Calibrate the alignment score against physical misalignment in Angstroms."""
    xml_files = sorted(Path().glob(tilt_series))
    if not xml_files:
        # also accept an absolute-path glob
        pattern = Path(tilt_series)
        xml_files = sorted(pattern.parent.glob(pattern.name))
    if not xml_files:
        raise typer.BadParameter(f"No XML files matched: {tilt_series}")

    output_directory.mkdir(parents=True, exist_ok=True)

    delta_steps = DEFAULT_DELTA_STEPS_PIXELS
    if max_delta_pixels is not None:
        if max_delta_pixels <= 0:
            raise typer.BadParameter("--max-delta-pixels must be positive")
        scale = max_delta_pixels / max(DEFAULT_DELTA_STEPS_PIXELS)
        delta_steps = tuple(d * scale for d in DEFAULT_DELTA_STEPS_PIXELS)

    model = load_model(model_checkpoint, device)

    print(f"\nCalibrating {len(xml_files)} tilt-series against {model_checkpoint}")
    print(f"  injected shifts (pixels): {', '.join(f'{d:g}' for d in delta_steps)}")
    print(f"  {max_positions} positions x {n_repeats} repeats per shift\n")

    results = []
    for i, xml_path in enumerate(xml_files, start=1):
        print(f"[{i}/{len(xml_files)}] {xml_path.stem} ...", flush=True)
        result = calibrate_one(
            model=model,
            xml_path=xml_path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            downsample=downsample,
            device=device,
            max_positions=max_positions,
            n_repeats=n_repeats,
            delta_steps_pixels=delta_steps,
            seed=seed,
        )
        results.append(result)

    report_path = output_directory / "score_calibration.json"
    with open(report_path, "w") as f:
        json.dump(
            {"model_checkpoint": str(model_checkpoint), "series": results}, f, indent=2
        )

    plot_path = output_directory / "score_calibration.png"
    plot_curves(results, plot_path)

    # Ranking by estimated residual error is the point of the exercise: unlike
    # the raw score, this column is in Angstroms and comparable across series.
    print(
        f"\n{'tilt-series':<34} {'score(0)':>10} {'resid (Å)':>12} "
        f"{'±':>8} {'R²':>7}  note"
    )
    print("-" * 100)
    for result in sorted(
        results,
        key=lambda x: (
            x["residual_error_angstrom"]
            if x["residual_error_angstrom"] is not None
            else float("inf")
        ),
    ):
        r = result["residual_error_angstrom"]
        err = result["residual_error_stderr"]
        r2 = result["fit_r_squared"]
        print(
            f"{result['tilt_series']:<34} {result['score_at_zero']:>10.4f} "
            f"{(f'{r:.2f}' if r is not None else 'n/a'):>12} "
            f"{(f'{err:.2f}' if err is not None else 'n/a'):>8} "
            f"{(f'{r2:.3f}' if r2 is not None else 'n/a'):>7}  {result['fit_note']}"
        )

    print(f"\nWrote {report_path}")
    print(f"Wrote {plot_path}\n")


if __name__ == "__main__":
    typer.run(main)
