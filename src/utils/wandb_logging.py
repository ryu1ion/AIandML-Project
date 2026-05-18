"""W&B logging (offline-first) for Phase 2.

PHASE2 hard-constraint #3 mandates W&B (project ``label-free-distill-phase2``).
This environment has no W&B credentials, so the default mode is ``offline``:
runs are written under ``wandb/`` and can be ``wandb sync``'d later once a key
is available. The trainer's local CSV/JSON logging is retained as a backup.

All W&B calls are best-effort: a logging failure must never kill a training
run, so every call is guarded.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_PROJECT = "label-free-distill-phase2"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


class WandbLogger:
    """Thin, fail-safe wrapper around ``wandb`` (no-op when disabled)."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str = DEFAULT_PROJECT,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        mode: str = "offline",
        entity: str | None = None,
        out_dir: str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.run = None
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb
        except Exception as e:  # pragma: no cover
            print(f"[wandb] import failed ({e!r}); continuing without W&B", flush=True)
            self.enabled = False
            return
        self._wandb = wandb
        os.environ.setdefault("WANDB_SILENT", "true")
        cfg = dict(config or {})
        cfg["git_commit"] = get_git_commit()
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        try:
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=cfg,
                mode=mode,
                entity=entity,
                dir=str(out_dir) if out_dir is not None else None,
                reinit=True,
                settings=wandb.Settings(console="off"),
            )
            print(
                f"[wandb] {mode} run '{run_name}' in project '{project}' "
                f"-> {getattr(self.run, 'dir', '?')}",
                flush=True,
            )
        except Exception as e:  # pragma: no cover
            print(f"[wandb] init failed ({e!r}); continuing without W&B", flush=True)
            self.enabled = False
            self.run = None

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self._wandb.log(data, step=step)
        except Exception:
            pass

    def summary(self, data: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            for k, v in data.items():
                self.run.summary[k] = v
        except Exception:
            pass

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self._wandb.finish()
        except Exception:
            pass
        self.run = None

    @property
    def run_dir(self) -> str | None:
        return getattr(self.run, "dir", None) if self.run is not None else None
