"""Append completed experiments without replacing earlier results (JSON Lines)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def append_result(path, *, args, model_config, data_config, history, started_at,
                  elapsed_seconds, device, parameter_count, start_epoch):
    if not history:
        raise ValueError("Cannot export a completed run without epoch metrics")
    best = max(history, key=lambda row: row["eval"]["accuracy"])
    record = {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {"args": args, "model": model_config, "data": data_config},
        "summary": {"status": "completed", "device": device, "parameter_count": parameter_count,
                    "elapsed_seconds_this_invocation": elapsed_seconds,
                    "start_epoch_this_invocation": start_epoch,
                    "epochs_trained_this_invocation": sum(row["epoch"] >= start_epoch for row in history),
                    "best_epoch": best["epoch"], "best_val_accuracy_percent": best["eval"]["accuracy"],
                    "final_epoch": history[-1]["epoch"],
                    "final_train": history[-1]["train"], "final_val": history[-1]["eval"],
                    "accuracy_unit": "percent"},
        "history": history,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(line)
