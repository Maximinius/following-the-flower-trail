#!/usr/bin/env python3
"""Run the full analysis pipeline.

Each run writes into its own folder, ``results/vN/`` by default, together with
a ``MANIFEST.md`` recording the timestamp, git commit, interpreter and the
outcome of every script. ``results/LATEST`` names the most recent version.

The main analysis runs with OUTLIER_HANDLING=none (no exclusion, no clipping). The model scripts in
``REPLICATED`` are then rerun with OUTLIER_HANDLING=mahalanobis into
``vN/mahalanobis/``, and ``replication_grid`` compares the two.

Configuration
    PIPELINE_VERSION   'auto' for the next unused vN, or a fixed label
    PIPELINE_SKIP      script names to skip
    RESULTS_DIR        (environment) explicit output folder; disables versioning

Every script needs R with lme4 and lmerTest.

Scripts, in dependency order:
    modality_lmm               H1/H2 modality contrasts (by-participant modality slope); Figure 2
    figure1                    Figure 1
    bayesian_modality          descriptives and Bayesian comparisons
    maze_lmm                   H3 (interaction models with a modality slope, in cognitive_lmm)
    flanker_lmm                H4
    gradcpt_lmm                H5
    wcst_lmm                   H6
    joint_lmm                  cross-domain joint model
    joint_lmm_exploratory      cognition x modality on the digital outcomes (modality slope, difference-score sensitivity)
    standalone_modality_lmm    within-modality models; Figures 3 and 4
    spatial_difficulty         H7 (segment class and modality slopes)
    outlier_multiverse         modality contrasts and H3 to H6 under five outlier treatments; Figure S3
    participant_level_checks   reliability, cross-modality correlations against the pre-registered power analysis,
                               speed-accuracy and redundancy on participant means; Figure S1
    replication_grid           main results against the Mahalanobis rerun; Figure S2
    si_tables                  Tables S1 to S10
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = Path(__file__).resolve().parent

SCRIPTS = [
    'modality_lmm',
    'figure1',
    'bayesian_modality',
    'maze_lmm',
    'flanker_lmm',
    'gradcpt_lmm',
    'wcst_lmm',
    'joint_lmm',
    'joint_lmm_exploratory',
    'standalone_modality_lmm',
    'spatial_difficulty',
    'outlier_multiverse',
    'participant_level_checks',
]

# Rerun with OUTLIER_HANDLING=mahalanobis into vN/mahalanobis/.
REPLICATED = [
    'modality_lmm',
    'bayesian_modality',
    'maze_lmm',
    'flanker_lmm',
    'gradcpt_lmm',
    'wcst_lmm',
    'joint_lmm',
    'joint_lmm_exploratory',
    'standalone_modality_lmm',
    'spatial_difficulty',
]
FINAL = ['replication_grid', 'si_tables']

PIPELINE_OUTPUT_DIR = 'results'
PIPELINE_VERSION = 'auto'
PIPELINE_SKIP: list[str] = []


def run_script(name: str, results_dir: str, handling: str = 'none') -> tuple[bool, float]:
    env = {**os.environ, 'RESULTS_DIR': results_dir, 'OUTLIER_HANDLING': handling}
    t0 = time.monotonic()
    result = subprocess.run([sys.executable, str(ANALYSIS / f'{name}.py')], env=env, cwd=str(ANALYSIS))
    return result.returncode == 0, time.monotonic() - t0


def next_version(parent: Path) -> str:
    used = [int(m.group(1)) for d in parent.glob('v*') if d.is_dir()
            for m in [re.match(r'^v(\d+)$', d.name)] if m]
    return f'v{max(used, default=0) + 1}'


def resolve_output_dir() -> tuple[Path, str | None]:
    """(output folder, version label). An explicit RESULTS_DIR turns versioning off."""
    explicit = os.environ.get('RESULTS_DIR', '').strip()
    if explicit:
        out = Path(explicit)
        out = out if out.is_absolute() else REPO_ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        return out, None
    parent = REPO_ROOT / PIPELINE_OUTPUT_DIR
    parent.mkdir(parents=True, exist_ok=True)
    label = os.environ.get('PIPELINE_VERSION', PIPELINE_VERSION).strip() or 'auto'
    if label == 'auto':
        label = next_version(parent)
    out = parent / label
    out.mkdir(parents=True, exist_ok=True)
    return out, label


def git_commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def write_manifest(out: Path, label: str | None, results, skipped) -> None:
    lines = [
        f'# Pipeline run {label or out.name}',
        '',
        f'- Run at: {datetime.now().astimezone().isoformat(timespec="seconds")}',
        f'- Git commit: {git_commit()}',
        f'- Python: {sys.executable}',
        f'- Output: {out}',
        '',
        '| Script | Outcome | Seconds |',
        '|---|---|---|',
    ]
    lines += [f'| `{name}` | {"ok" if ok else "FAILED"} | {elapsed:.1f} |' for name, ok, elapsed in results]
    if skipped:
        lines += ['', 'Skipped: ' + ', '.join(f'`{s}`' for s in skipped)]
    lines += ['', 'Regenerate with `analysis/run_pipeline.py`.', '']
    (out / 'MANIFEST.md').write_text('\n'.join(lines), encoding='utf-8')
    if label:
        (out.parent / 'LATEST').write_text(label + '\n', encoding='utf-8')


def main() -> None:
    out, label = resolve_output_dir()
    env_skip = os.environ.get('PIPELINE_SKIP', '').strip()
    skip = [s.strip() for s in env_skip.split(',') if s.strip()] if env_skip else list(PIPELINE_SKIP)
    replication = out / 'mahalanobis'
    to_run = ([(s, out, 'none', s) for s in SCRIPTS if s not in skip]
              + [(s, replication, 'mahalanobis', f'{s} (mahalanobis)') for s in REPLICATED if s not in skip]
              + [(s, out, 'none', s) for s in FINAL if s not in skip])
    skipped = [s for s in SCRIPTS + FINAL if s in skip]

    print(f'Version : {label or "(unversioned, RESULTS_DIR set)"}')
    print(f'Output  : {out}')
    print(f'Python  : {sys.executable}')
    if skipped:
        print(f'Skipped : {", ".join(skipped)}')
    print()

    results = []
    for idx, (script, folder, handling, name) in enumerate(to_run, start=1):
        print(f'[{idx}/{len(to_run)}] {name}', flush=True)
        folder.mkdir(parents=True, exist_ok=True)
        ok, elapsed = run_script(script, str(folder), handling)
        print(f'      {"ok" if ok else "FAILED"} ({elapsed:.1f}s)\n', flush=True)
        results.append((name, ok, elapsed))

    print('=' * 50)
    for name, ok, elapsed in results:
        print(f'  {"ok    " if ok else "FAILED"} {name:<42} {elapsed:6.1f}s')
    print(f'\n  Total: {sum(e for _, _, e in results):.1f}s  |  '
          f'{sum(ok for _, ok, _ in results)}/{len(results)} succeeded')
    write_manifest(out, label, results, skipped)
    print(f'  Manifest: {out / "MANIFEST.md"}')

    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f'\nFailed: {", ".join(failed)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
