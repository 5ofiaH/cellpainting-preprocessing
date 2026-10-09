"""
Run Summary: a traceable, always-up-to-date record of each pipeline step.

Writes both a human-readable Markdown file and a machine-readable JSON, updated
after every step so you can trace how many rows/sites/wells/features entered and
left each stage, how many were dropped, and why.

Usage
-----
    summary = RunSummary(out_dir, cell_line)
    summary.add_step("Data loading", metrics={...}, notes="...")
    summary.set_meta(project="CP-MS-BA", config="config_x.yaml")
    # file is rewritten on every add_step / set_meta call
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class RunSummary:
    """Accumulates per-step metrics and writes an auto-updating summary file.

    ``out_dir`` is the directory the summary files are written into directly
    (typically the per-run features folder, e.g.
    ``<base_dir>/<cell_line>/<features_subdir>``, so a dated
    ``features_subdir`` keeps a rerun's summary from overwriting an older
    run's) — NOT a base directory that ``cell_line`` gets appended to.
    """

    def __init__(self, out_dir: str, cell_line: str, resume: bool = False):
        """``resume=True`` continues an existing run_summary.json (a pipeline
        run restarted from a later step) instead of starting a new one."""
        self.cell_line = cell_line
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.md_path = self.dir / "RUN_SUMMARY.md"
        self.json_path = self.dir / "run_summary.json"
        self.meta: Dict[str, Any] = {"cell_line": cell_line}
        self.steps: List[Dict[str, Any]] = []
        self._started = self._now()
        if resume and self.json_path.exists():
            try:
                prev = json.loads(self.json_path.read_text(encoding="utf-8"))
                self.meta.update(prev.get("meta", {}))
                self.steps = list(prev.get("steps", []))
                self._started = prev.get("started", self._started)
            except Exception:
                pass

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def set_meta(self, **kwargs) -> None:
        """Set top-level metadata (project, config path, etc.)."""
        self.meta.update(kwargs)
        self._write()

    def add_step(self, name: str, metrics: Optional[Dict[str, Any]] = None,
                 notes: str = "") -> None:
        """Record one pipeline step and rewrite the summary files."""
        self.steps.append({
            "step": name,
            "timestamp": self._now(),
            "metrics": {k: _clean(v) for k, v in (metrics or {}).items()},
            "notes": notes,
        })
        self._write()

    # ── writers ──────────────────────────────────────────────────────────────
    def _write(self) -> None:
        self._write_json()
        self._write_md()

    def _write_json(self) -> None:
        payload = {
            "meta": self.meta,
            "started": self._started,
            "updated": self._now(),
            "steps": self.steps,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def _write_md(self) -> None:
        lines: List[str] = []
        lines.append(f"# Run Summary — {self.cell_line}")
        lines.append("")
        lines.append(f"_Started {self._started} · updated {self._now()}_")
        lines.append("")

        if self.meta:
            lines.append("## Configuration")
            lines.append("")
            for k, v in self.meta.items():
                if k == "cell_line":
                    continue
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        lines.append("## Pipeline steps")
        lines.append("")
        for i, step in enumerate(self.steps, 1):
            lines.append(f"### {i}. {step['step']}")
            lines.append(f"_{step['timestamp']}_")
            lines.append("")
            metrics = step.get("metrics") or {}
            if metrics:
                lines.append("| Metric | Value |")
                lines.append("|---|---|")
                for k, v in metrics.items():
                    lines.append(f"| {k} | {_fmt(v)} |")
                lines.append("")
            if step.get("notes"):
                lines.append(f"> {step['notes']}")
                lines.append("")

        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def _clean(v: Any) -> Any:
    """Coerce numpy scalars to native python for JSON."""
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except ImportError:
        pass
    return v


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)
