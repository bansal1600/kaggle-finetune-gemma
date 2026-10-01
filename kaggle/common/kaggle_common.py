"""Code shared by our Kaggle notebooks (grader check, agent eval).

Kaggle runs a single script file, so scripts/kaggle_build.py pastes this module into each notebook
in place of its `from kaggle_common import ...` line. For local runs, put kaggle/common on PYTHONPATH.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

# GPU-only packages: not needed when no model is served.
SKIP_WHEELS = ("vllm-", "flashinfer", "bitsandbytes", "xgrammar", "nvidia_", "cutlass")


def install_harness(wheelhouse: Path, gpu: bool = False) -> None:
    """Install the organizers' harness packages, the same way their getting-started notebook does.

    With gpu=False the GPU-only packages (vLLM, CUDA kernels) are skipped.
    """
    if not wheelhouse.is_dir():
        print(f"No wheelhouse at {wheelhouse}; assuming the harness is already installed.")
        return
    tmp = Path("/tmp/wheelhouse")
    tmp.mkdir(parents=True, exist_ok=True)
    for whl in wheelhouse.glob("*.whl"):
        if not gpu and whl.name.startswith(SKIP_WHEELS):
            continue
        # Kaggle strips '+' from wheel names on upload; restore the PEP 440 local version tag.
        name = whl.name.replace("cu128", "+cu128") if "cu128" in whl.name and "+" not in whl.name else whl.name
        if not (tmp / name).exists():
            os.symlink(whl, tmp / name)
    wheels = sorted(str(w) for w in tmp.glob("*.whl"))
    print(f"Installing {len(wheels)} harness wheels...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *wheels], check=True)
    importlib.invalidate_caches()


def use_scorer_environment(wheels_dir: Path, target: Path) -> None:
    """Give every task sandbox the same Python packages the real scorer gives it.

    The scorer's Docker sandbox unpacks, from the competition's wheels/ folder, the highest
    version of each package and takes pytest from its image. In notebook ("subprocess") mode the
    harness skips that and lets tasks import whatever the notebook happens to have installed, so
    a task could pass or fail locally for reasons unrelated to the code. Here we:
      1. install the scorer's package set (highest compatible version of each wheel) into `target`,
      2. make each task's private venv see only that set, not the notebook's packages,
      3. run the editable install and sandbox/setup.py step the scorer runs.
    Known differences left: Python 3.12 here vs 3.13 in the scorer, and pytest is the newest one in
    wheels/ rather than the scorer image's own copy.
    """
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename

    import swegemma.harness.container_setup as cs
    import swegemma.harness.verification as ver
    from swegemma.sandbox.subprocess import SubprocessManager

    supported = set(sys_tags())
    best: dict[str, tuple] = {}
    for whl in wheels_dir.glob("*.whl"):
        try:
            name, version, _, tags = parse_wheel_filename(whl.name)
        except Exception:
            continue
        if tags & supported and (name not in best or version > best[name][0]):
            best[name] = (version, whl)
    if not (target / ".done").exists():
        print(f"Installing the scorer's package set ({len(best)} packages) into {target}...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index",
                        "--target", str(target), *[str(w) for _, w in best.values()]], check=True)
        (target / ".done").write_text("\n".join(sorted(f"{n}=={v}" for n, (v, _) in best.items())))

    original_start = SubprocessManager.start

    def start(self):
        self.system_site_packages = False
        sandbox_id = original_start(self)
        for site_dir in self._sandboxes[sandbox_id]["venv"].glob("lib/python*/site-packages"):
            # Named to load after the editable install's .pth, so the task's own repo wins over
            # the copy of fastapi/rich/requests/httpx that is also in the scorer's package set.
            (site_dir / "zz_scorer_env.pth").write_text(f"{target}\n")
        return sandbox_id

    SubprocessManager.start = start

    def install_editable_package(docker, container_id):
        docker.exec(container_id, "pip install --no-index --find-links=/wheels --no-build-isolation "
                                  "--no-deps -e /workspace 2>/dev/null || true")

    def install_test_dependencies(docker, container_id, repo="", *, fast_path=True, config=None):
        script = cs.resolve_sandbox_setup_script(config)
        if script is None:
            return
        docker.copy_to(container_id, script, "/tmp/setup.py")
        docker.exec(container_id, f"python3 /tmp/setup.py --fast-path {repo}".strip())
        docker.exec(container_id, "rm -f /tmp/setup.py")

    for module in (cs, ver):
        module.install_editable_package = install_editable_package
        module.install_test_dependencies = install_test_dependencies
