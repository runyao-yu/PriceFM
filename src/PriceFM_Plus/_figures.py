"""Regenerate the paper's grouped figures from the released plot inputs."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


VARIANTS = (
    "single_chronological_branch", "no_lag_lead",
    "no_temporal_mixing", "no_region_encoding",
)


def regenerate(archive: Path, output: Path) -> Path:
    from ._api import _extract

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pricefm_figures_") as temp:
        source = _extract(archive, Path(temp) / "source")

        def run(script: str, *args: str) -> None:
            command = [sys.executable, str(source / script), *args]
            completed = subprocess.run(command, cwd=source, text=True,
                                       capture_output=True, check=False)
            if completed.returncode:
                tail = (completed.stdout + "\n" + completed.stderr)[-4000:]
                raise RuntimeError(f"Figure step {script} {args} failed:\n{tail}")

        run("eda_group_build.py", "surf", "0.676", "-7", "-11.5", "-9.5")
        run("eda_group_build.py", "map", "0.815")
        run("eda_group_build.py", "crop", "0", "SURF")
        run("eda_group_build.py", "crop", "0", "MAP")
        run("eda_group_build.py", "week", "163.58")
        run("plot_eda_group.py")

        for model in ("MLP", "LSTM", "Transformer", "PriceFM+"):
            run("jt_group_build.py", "map", model, "0.809")
        run("jt_group_build.py", "crop")
        run("jt_group_build.py", "small", "38.57")
        run("plot_jt_group.py")

        run("fm_group_build.py", "maps", "0.809")
        run("fm_group_build.py", "cropmaps")
        run("fm_group_build.py", "small", "37.77")
        run("plot_fm_group.py")

        run("cr_plots.py")
        run("abl_plots.py")

        figures = source / "figures"
        targets = {
            "eda_figure.pdf": figures / "EDA_GROUP_GREENRED" / "eda_figure.pdf",
            "jt_figure.pdf": figures / "JOINT_COMPARE" / "jt_figure.pdf",
            "cr_figure.pdf": figures / "CRISIS" / "cr_figure.pdf",
            "fm_figure.pdf": figures / "FM_COMPARE" / "fm_figure.pdf",
            "abl_figure.pdf": figures / "ABLATION" / "abl_figure.pdf",
        }
        for variant in VARIANTS:
            targets[f"model_ablation_{variant}.pdf"] = (
                figures / "ABLATION" / "BARS" / f"bars_{variant}.pdf"
            )
        missing = [str(path) for path in targets.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"Figure generation missed files: {missing}")
        for name, path in targets.items():
            shutil.copy2(path, output / name)
    return output
