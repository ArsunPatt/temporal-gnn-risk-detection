"""Execute the frozen viewer against a public release with exact manifest membership."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from scripts.verify_release import MANIFEST, SIDECAR, digest, release_members

NOTEBOOK = Path("notebooks/05_research_results.ipynb")
SUMMARY = "results/stage5/execution_summary.json"


def snapshot(root):
    """Catch byte changes, identical rewrites, and extra or missing files."""
    return {
        name: (digest(root / name), (root / name).stat().st_mtime_ns)
        for name in release_members(root, checkout=False)
    }


def execute_viewer(root, runtime):
    """Run a fresh kernel from this test's interpreter and keep outputs in memory."""
    kernel_dir = runtime / "publication-test"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Publication test",
        "language": "python",
        "env": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "IPYTHONDIR": str(runtime / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(runtime),
        },
    }))
    specs = KernelSpecManager(kernel_dirs=[str(runtime)], ensure_native_kernel=False)
    kernel = KernelManager(kernel_name="publication-test", kernel_spec_manager=specs,
                           connection_file=str(runtime / "kernel.json"))
    notebook = nbformat.read(root / NOTEBOOK, as_version=4)
    return NotebookClient(
        notebook, km=kernel, timeout=60, resources={"metadata": {"path": str(root)}},
    ).execute()


def test_notebook05_executes_without_writing_release_files(tmp_path):
    source = Path.cwd()
    public = tmp_path / "public"
    manifest = json.loads((source / MANIFEST).read_text())
    for name in {*manifest["public_files_sha256"], MANIFEST.as_posix(), SIDECAR.as_posix()}:
        path = public / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, path)

    before = snapshot(public)
    notebook = execute_viewer(public, tmp_path / "runtime")
    verified = subprocess.run(
        [sys.executable, "scripts/verify_release.py"], cwd=public,
        capture_output=True, text=True, timeout=30,
    )
    after = snapshot(public)
    assert after[SUMMARY] == before[SUMMARY], "The historical execution summary was rewritten"
    assert after == before, "Notebook 05 modified the release file set, bytes, or timestamps"
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["file_membership_verified"]
    figures = [
        output for cell in notebook.cells for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    assert len(figures) == 19
