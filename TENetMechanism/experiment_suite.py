"""Versioned experimental presets and project locking (stdlib only)."""
import contextlib
import itertools
import os
from pathlib import Path

from experiment_data import fingerprint, read_json, atomic_json, load_manifest


def source_fingerprint():
    from experiment_data import digest_file
    root = Path(__file__).resolve().parent
    return fingerprint({p.name: digest_file(p) for p in sorted(root.glob("*.py"))})


@contextlib.contextmanager
def project_lock(project):
    """Kernel lock is automatically released on crashes; lock file may remain."""
    path = Path(project) / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    stream.seek(0, 2)
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise RuntimeError("This project is already running in another process") from error
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def load_project(project):
    project = Path(project).resolve()
    settings = read_json(project / "project.json")
    if settings["fingerprint"] != fingerprint({k: v for k, v in settings.items() if k != "fingerprint"}):
        raise ValueError("Project settings changed: create a new project instead of editing frozen settings")
    manifest = load_manifest(project / "manifest.json")
    if settings["manifest_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Wrong dataset manifest for this project")
    plan = read_json(project / "plan.json")
    if plan["fingerprint"] != fingerprint(plan["jobs"]):
        raise ValueError("Experiment plan was edited outside the planner")
    return project, settings, manifest, plan


def presets(settings, batch):
    width = settings["width"]
    base = dict(architecture="research", num_classes=settings["num_classes"],
        image_size=settings["image_size"], channels=width, operators=["G"] * 3,
        hidden=32, normalization="standardized", head_grid=settings["head_grid"],
        head_hidden=settings["head_hidden"], dropout=.1)
    def research(**changes):
        return {**base, **changes}
    full = dict(architecture="tenet", num_classes=settings["num_classes"], image_size=settings["image_size"],
        channels=width, layers=3, alpha=0, head_grid=settings["head_grid"], head_hidden=settings["head_hidden"],
        dropout=.1, readout="topology", learnable_phi=True, trace=True, edge=True, directed=False,
        cross=True, chebyshev=False, spatial=False)
    p1 = {name.replace("+", "plus"): research(operators=[name] * 3) for name in ("P", "V", "Q", "E", "G", "G+")}
    p1["FullS"] = full
    positions = {"".join(ops): research(operators=list(ops)) for ops in itertools.product(("G", "Q"), repeat=3)
                 if len(set(ops)) > 1}
    norms = {f"G_{norm}": research(normalization=norm) for norm in ("centered", "raw")}
    if batch == "core":
        return {"Reference": {**full, "channels": 3, "alpha": 8}, **p1, **positions, **norms}
    if batch == "p1":
        return p1
    if batch == "positions":
        return {"Q": p1["Q"], "G": p1["G"], **positions}
    if batch == "normalization":
        return {"G": p1["G"], **norms}
    if batch == "topology":
        return {"G": p1["G"], "outer": research(operators=["outer"] * 3),
                **{f"mismatch{i}": research(operators=["mismatch"] * 3, mismatch_seed=i) for i in (1, 2, 3)},
                **{name: research(operators=[name] * 3) for name in ("Z", "ZQ")}}
    if batch == "depth":
        return {f"{op}_L{depth}" if depth != 3 else op: research(operators=[op] * depth)
                for op in ("G", "Q") for depth in (1, 2, 3, 4)}
    if batch == "capacity":
        return {f"{op}_C{width * 2}": research(operators=[op] * 3, channels=width * 2) for op in ("G", "Q")}
    if batch == "residual":
        return {f"{op}_residual": research(operators=[op] * 3, residual=True) for op in ("G", "Q")}
    if batch == "schedule":
        return {f"{op}_{schedule}": research(operators=[op] * 3, schedule=schedule)
                for op in ("G", "Q") for schedule in ("DDDPPP", "DPDPDP")}
    if batch == "budget":
        return {**{f"G_C{c}": research(channels=c) for c in (8, 16, 32)},
                **{f"Q_C{c}_H{h}": research(operators=["Q"] * 3, channels=c, hidden=h)
                   for c in (8, 16, 24, 32) for h in (16, 32, 64, 128)}}
    raise ValueError(f"Unknown experimental batch: {batch}")


BATCHES = ("core", "p1", "positions", "normalization", "topology", "depth", "capacity", "residual", "schedule", "budget", "confirm")


def add_jobs(project, batch="core", seeds=(42,), models=None):
    project, settings, manifest, plan = load_project(project)
    if (project / "final_registration.json").exists():
        raise ValueError("Final comparisons already frozen; use a new project for further exploration")
    if batch == "confirm":
        if not models:
            raise ValueError("confirm requires --models with the preselected core comparisons")
        choices = {j["model_id"]: j["model_config"] for j in plan["jobs"]}
    else:
        choices = presets(settings, batch)
    choices = validate_choices(settings, choices, models)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Provide distinct nonnegative training seeds")
    existing = {j["run_id"]: j for j in plan["jobs"]}
    new = []
    for name, config in choices.items():
        for seed in seeds:
            job = dict(run_id=f"{name}_s{seed}", model_id=name, seed=seed, batch=batch, model_config=config)
            job["fingerprint"] = fingerprint({"model": config, "seed": seed, "project": settings["fingerprint"]})
            if job["run_id"] in existing:
                if existing[job["run_id"]]["fingerprint"] != job["fingerprint"]:
                    raise ValueError("A run id cannot be reused for different configurations")
            else:
                new.append(job)
    plan["jobs"].extend(new)
    plan["fingerprint"] = fingerprint(plan["jobs"])
    atomic_json(project / "plan.json", plan)
    return new


def validate_choices(settings, choices, models=None):
    """Validate the entire batch before any project artifacts are committed."""
    if models:
        missing = set(models) - set(choices)
        if missing:
            raise ValueError(f"Unknown models: {sorted(missing)}; available: {sorted(choices)}")
        choices = {name: choices[name] for name in models}
    for name, config in choices.items():
        depth = len(config["operators"]) if "operators" in config else config["layers"]
        if settings["image_size"] % 2 ** depth or settings["image_size"] // 2 ** depth < settings["head_grid"]:
            raise ValueError(f"{name}: incompatible image size/head grid; create a new appropriate project")
    return choices
