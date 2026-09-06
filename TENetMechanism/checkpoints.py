"""Resolve checkpoint paths without silently replacing an explicit experiment."""
from pathlib import Path


def resolve_checkpoint(value=None, runs_dir=None):
    root = Path(runs_dir) if runs_dir is not None else Path(__file__).resolve().parent / "runs"
    candidates = sorted((p for p in root.rglob("last.pth") if p.is_file()),
                        key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
    if value is None or value == "latest":
        if candidates:
            return candidates[0].resolve()
        raise FileNotFoundError(f"No last.pth found under {root.resolve()}. Train first or pass --checkpoint /path/to/last.pth")
    target = Path(value).expanduser()
    if target.is_dir():
        target = target / "last.pth"
    if target.is_file():
        return target.resolve()
    available = "\n".join(f"  {p.resolve()}" for p in candidates[:10]) or "  (none)"
    raise FileNotFoundError(f"Checkpoint not found: {target.resolve()}\nAvailable last.pth files:\n{available}\n"
                            "Pass one of these paths, or omit --checkpoint to use the latest run.")
