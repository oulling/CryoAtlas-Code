"""CryoAtlas addition: optional W&B logging for CNN training.

Adds a SummaryWriter-compatible logger, explicit training-step axes, and
graceful operation when online logging is unavailable.

We use ``wandb.define_metric`` to declare an explicit integer step axis
("global_step") so that our training/validation metrics are tied to the
training step count rather than wandb's implicit sequential step counter
(which is driven by system metric uploads and would otherwise cause our
explicit-step log calls to be silently dropped).
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional


class _NullWriter:
    def __init__(self, reason: str = ""):
        if reason:
            print(f"[summary_writer_wrapper] wandb disabled: {reason}")

    def log(self, *a, **kw): pass
    def add_scalar(self, *a, **kw): pass
    def add_image(self, *a, **kw): pass
    def flush(self): pass
    def close(self): pass


class SummaryWriterWrapper:
    def __init__(self, log_dir: str, debug: bool = False,
                 project: Optional[str] = None, run_name: Optional[str] = None,
                 config: Optional[dict] = None):
        os.makedirs(log_dir, exist_ok=True)
        self.debug = debug
        self._wandb: Any = None
        self.writer: Any = _NullWriter()

        try:
            import wandb  # type: ignore
        except ImportError:
            self.writer = _NullWriter("wandb not installed (pip install wandb)")
            return

        if not os.environ.get("WANDB_API_KEY") and not os.path.exists(
            os.path.expanduser("~/.netrc")
        ):
            self.writer = _NullWriter(
                "no WANDB_API_KEY in env and no ~/.netrc; run `wandb login` or export WANDB_API_KEY"
            )
            return

        project = project or os.environ.get("WANDB_PROJECT", "modelangelo-stage1")
        run_name = run_name or os.environ.get("WANDB_RUN_NAME") or os.path.basename(
            os.path.abspath(log_dir)
        )

        try:
            self._wandb = wandb
            self.writer = wandb.init(
                project=project,
                name=run_name,
                dir=log_dir,
                config=config or {},
                resume="allow",
            )
            # Define an explicit integer step axis so train/valid charts use
            # the training step as x rather than wandb's implicit step.
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("valid/*", step_metric="global_step")
            print("[summary_writer_wrapper] wandb run started:",
                  getattr(self.writer, "url", "(no url)"), flush=True)
        except Exception as e:
            print(f"[summary_writer_wrapper] wandb.init failed: {e}", flush=True)
            self.writer = _NullWriter(f"wandb.init failed: {e}")
            self._wandb = None

    # SummaryWriter-compatible API -----------------------------------------
    def _log(self, payload: dict, step: Optional[int]):
        if self._wandb is None:
            print("[summary_writer_wrapper] _wandb is None; cannot log "
                  f"payload={list(payload.keys())}", flush=True)
            return
        data = dict(payload)
        if step is not None:
            data["global_step"] = int(step)
        if not getattr(self, "_logged_once", False):
            print(f"[summary_writer_wrapper] first log call: keys={list(data.keys())}",
                  flush=True)
            self._logged_once = True
        try:
            self._wandb.log(data)
        except Exception as e:  # never crash the training loop for logging
            # Always print — this was previously gated behind debug and hid bugs.
            print(f"[summary_writer_wrapper] wandb.log({list(data.keys())}) failed: "
                  f"{e!r}", flush=True)

    def add_scalar(self, tag: str, value: float, step: Optional[int] = None):
        self._log({tag: float(value)}, step)
        if self.debug:
            print(f"[wandb] {tag} = {float(value):.5f} @ {step}")

    def add_scalars(self, main_tag: str, tag_scalar_dict: Mapping[str, float],
                    step: Optional[int] = None):
        payload = {f"{main_tag}/{k}": float(v) for k, v in tag_scalar_dict.items()}
        self._log(payload, step)

    def add_image(self, tag: str, img, step: Optional[int] = None):
        if self._wandb is not None:
            self._wandb.log({tag: self._wandb.Image(img),
                             **({"global_step": int(step)} if step is not None else {})})

    def log(self, payload: Mapping[str, float], step: Optional[int] = None):
        """Native wandb-style bulk log of many metrics at once."""
        self._log(dict(payload), step)

    def flush(self):
        pass

    def close(self):
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
