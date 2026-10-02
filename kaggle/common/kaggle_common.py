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


def find_extra_wheels(expected: Path | None, marker: str = "annotated_doc-*.whl") -> Path | None:
    """Locate our extra-wheels dataset; fail loudly instead of silently grading without it.

    Kaggle's mount path for attached datasets has changed before (grader check v2 did not find it
    at the expected path and quietly ran without the 7 packages), so search /kaggle/input as well.
    """
    if expected is None or (expected.is_dir() and any(expected.glob(marker))):
        return expected
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for hit in sorted(kaggle_input.rglob(marker)):
            print(f"Extra wheels not at {expected}; using {hit.parent}", flush=True)
            return hit.parent
        tree = sorted(str(p) for p in kaggle_input.glob("*/*"))
        raise SystemExit(f"Extra wheels ({marker}) not found under /kaggle/input. Mounted: {tree}")
    raise SystemExit(f"Extra wheels not found at {expected}")


def make_workspace_editable(site_dir: Path, workspace: Path) -> None:
    """Turn pip's copy of the task repo into an editable install, as the scorer's `pip install -e` is.

    pip --target copies the repo at its base commit into site-packages. For src/ layouts (requests)
    that copy shadowed /workspace/src: PYTHONPATH=/workspace does not reach src/, so the agent's
    repro scripts and the hidden tests ran against the unedited copy. Remove the copied files (keep
    the .dist-info metadata) and point a .pth file at the live source instead. "__editable__" sorts
    before "__scorer_env", so the workspace also wins over the package set's release of the repo.
    """
    import json
    from urllib.parse import unquote, urlparse

    for dist_info in site_dir.glob("*.dist-info"):
        direct_url = dist_info / "direct_url.json"
        record = dist_info / "RECORD"
        if not (direct_url.exists() and record.exists()):
            continue
        url = json.loads(direct_url.read_text()).get("url", "")
        if Path(unquote(urlparse(url).path)).resolve() != workspace.resolve():
            continue
        for line in record.read_text().splitlines():
            rel = line.split(",", 1)[0]
            path = (site_dir / rel).resolve()
            if rel and site_dir.resolve() in path.parents and dist_info.resolve() not in path.parents:
                path.unlink(missing_ok=True)
                parent = path.parent
                while parent != site_dir.resolve() and parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
    src = workspace / "src"
    (site_dir / "__editable__workspace.pth").write_text(f"{src if src.is_dir() else workspace}\n")


def use_scorer_environment(wheels_dir: Path, target: Path, extra_wheels: Path | None = None) -> None:
    """Give every task sandbox the same Python packages the real scorer gives it.

    The scorer's Docker sandbox unpacks, from the competition's wheels/ folder, the highest
    version of each package and takes pytest from its image. In notebook ("subprocess") mode the
    harness skips that and lets tasks import whatever the notebook happens to have installed, so
    a task could pass or fail locally for reasons unrelated to the code. Here we:
      1. install the scorer's package set (highest compatible version of each wheel) into `target`,
         plus `extra_wheels`: packages the tests import that the public wheels/ folder lacks
         (annotated_doc, dirty_equals, typing_inspection, inline_snapshot, wrapt and deps),
      2. put that set first on each task's import path; the notebook's own packages stay last, as
         a fallback only,
      3. run the editable install and sandbox/setup.py step the scorer runs.
    Known differences left: Python 3.12 here vs 3.13 in the scorer, and pytest is the newest one in
    wheels/ rather than the scorer image's own copy.
    """
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename

    import swegemma.harness.container_setup as cs
    import swegemma.harness.verification as ver
    from swegemma.sandbox.subprocess import SubprocessManager

    extra_wheels = find_extra_wheels(extra_wheels)
    supported = set(sys_tags())
    best: dict[str, tuple] = {}
    wheels = list(wheels_dir.glob("*.whl"))
    competition_names = set()
    for whl in wheels:
        try:
            competition_names.add(parse_wheel_filename(whl.name)[0])
        except Exception:
            pass
    if extra_wheels and extra_wheels.is_dir():
        # Only fill gaps: never replace a version the competition's wheels/ folder provides.
        wheels += [w for w in extra_wheels.glob("*.whl") if parse_wheel_filename(w.name)[0] not in competition_names]
    for whl in wheels:
        try:
            name, version, _, tags = parse_wheel_filename(whl.name)
        except Exception:
            continue
        if tags & supported and (name not in best or version > best[name][0]):
            best[name] = (version, whl)
    if not (target / ".done").exists():
        print(f"Installing the scorer's package set ({len(best)} packages, extras from {extra_wheels}) "
              f"into {target}...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index",
                        "--target", str(target), *[str(w) for _, w in best.values()]], check=True)
        (target / ".done").write_text("\n".join(sorted(f"{n}=={v}" for n, (v, _) in best.items())))

    original_start = SubprocessManager.start

    def start(self):
        sandbox_id = original_start(self)
        for site_dir in self._sandboxes[sandbox_id]["venv"].glob("lib/python*/site-packages"):
            # .pth files load in name order. "__scorer_env" comes after "__editable__*" (so the
            # task's own repo wins over the fastapi/rich/requests/httpx copies in the package set)
            # and before the harness's "_host_env" (so the notebook's packages are only a fallback).
            (site_dir / "__scorer_env.pth").write_text(f"{target}\n")
        return sandbox_id

    SubprocessManager.start = start

    # Absolute host paths: in subprocess mode the harness copies wheels/ to /wheels/wheels/, so
    # "/wheels" itself is empty there (the Docker scorer is not affected).
    links = [f"--find-links={wheels_dir.resolve()}"]
    if extra_wheels and extra_wheels.is_dir():
        links.append(f"--find-links={extra_wheels.resolve()}")

    def install_editable_package(docker, container_id):
        # With dependencies (the scorer passes --no-deps): wheels/ holds many versions of e.g.
        # starlette so each repo snapshot can get the one its pyproject allows. They go into the
        # task's own venv, which comes before the shared package set on the import path.
        # Run from the notebook, not through docker.exec, because (found in grader check v3):
        #   - Kaggle's Python has no ensurepip, so task venvs have no pip of their own;
        #   - exec rewrites every /tmp/... and /usr/local/bin path in a command to sandbox paths.
        # --target also ignores what the notebook already has installed.
        # PYTHONPATH=<package set> lets pip find the repo's build backend (pdm-backend).
        paths = docker._sandboxes[container_id]
        site_dir = next(paths["venv"].glob("lib/python*/site-packages"))
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-index", *links,
             "--no-build-isolation", "--upgrade", "--target", str(site_dir), str(paths["workspace"])],
            env={**os.environ, "PYTHONPATH": str(target)}, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [{container_id[:12]}] repo dependency install failed: "
                  f"{result.stderr.strip().splitlines()[-1:] or result.returncode}", flush=True)
        make_workspace_editable(site_dir, paths["workspace"])

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
