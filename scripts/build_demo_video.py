"""Stitch the screenshots/*.png into a short MP4 walkthrough.

Uses matplotlib + matplotlib.animation (writer = ffmpeg). Falls back to
imageio-ffmpeg if a system ffmpeg is unavailable.

Run with: ``python scripts/build_demo_video.py``
Output: ``docs/demo_slideshow.mp4``
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "screenshots"
OUT = ROOT / "docs" / "demo_slideshow.mp4"
OUT.parent.mkdir(exist_ok=True)

CAPTIONS = {
    "01_eda_target_distribution.png": "EDA — class balance",
    "02_eda_correlation_heatmap.png": "EDA — feature correlation",
    "03_mlflow_experiments.png":      "MLflow — experiment leaderboard",
    "04_mlflow_run_details.png":      "MLflow — best run details",
    "05_pytest_passing.png":          "pytest — 11/11 passing",
    "06_github_actions_success.png":  "GitHub Actions — CI green",
    "07_docker_build.png":            "Docker — image build",
    "07_metrics_endpoint.png":        "API — /metrics (Prometheus)",
    "08_docker_run_predict.png":      "API — /predict response",
    "09_kubectl_get_all.png":         "Kubernetes — pods + service",
    "10_loadbalancer_endpoint.png":   "Deployed API URL",
    "11_swagger_ui.png":              "FastAPI — Swagger UI",
    "12_grafana_dashboard.png":       "Grafana — live dashboard",
    "13_prometheus_targets.png":      "Prometheus — targets up",
}


def main() -> int:
    images = sorted(p for p in SHOTS.glob("*.png"))
    if not images:
        print("ERROR: no screenshots found in", SHOTS); return 1

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0f111a")
    ax.axis("off")
    im = ax.imshow(imread(str(images[0])))
    title = ax.set_title("", color="white", fontsize=16, pad=12)

    def update(i: int):
        path = images[i]
        im.set_data(imread(str(path)))
        title.set_text(f"{i+1}/{len(images)}  —  {CAPTIONS.get(path.name, path.stem)}")
        return [im, title]

    a = anim.FuncAnimation(fig, update, frames=len(images), interval=2200, blit=False)

    try:
        writer = anim.FFMpegWriter(fps=1, bitrate=2000)
        a.save(str(OUT), writer=writer, dpi=120, savefig_kwargs={"facecolor": fig.get_facecolor()})
    except (FileNotFoundError, RuntimeError, ValueError):
        # Fall back to imageio-ffmpeg's bundled binary if available.
        try:
            import imageio_ffmpeg  # type: ignore
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            writer = anim.FFMpegWriter(fps=1, bitrate=2000)
            a.save(str(OUT), writer=writer, dpi=120, savefig_kwargs={"facecolor": fig.get_facecolor()})
        except Exception as exc:  # noqa: BLE001
            print("ffmpeg unavailable; install ffmpeg or `pip install imageio-ffmpeg`. Error:", exc)
            return 2

    plt.close(fig)
    print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
