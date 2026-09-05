"""Print the last four JSONL experiments as a Markdown table; standard library only."""

import argparse
import json
from collections import deque
from pathlib import Path


def load_tail(path, count=4):
    if count < 1:
        raise ValueError("count must be positive")
    with Path(path).open(encoding="utf-8-sig") as stream:
        lines = deque(((number, line) for number, line in enumerate(stream, 1) if line.strip()), maxlen=count)
    if not lines:
        raise ValueError("Results file is empty")
    records = []
    for number, line in lines:
        try:
            record = json.loads(line)
            if not record["history"] or record["summary"]["status"] != "completed":
                raise ValueError("expected a completed run with nonempty history")
            records.append(record)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid result at line {number}: {error}") from error
    return records


def render_table(records):
    lines = ["| Pruner | Best val % (epoch) | Final val % | Last <=10 epochs mean % | Final val loss | Total time (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    signatures = []
    names = []
    for record in records:
        config, summary = record["configuration"], record["summary"]
        args, model, data = config["args"], config["model"], config["data"]
        name = args["pruner"]
        names.append(name)
        tail = record["history"][-10:]
        average = sum(row["eval"]["accuracy"] for row in tail) / len(tail)
        lines.append(f"| {name} | {summary['best_val_accuracy_percent']:.2f} ({summary['best_epoch']}) | "
                     f"{summary['final_val']['accuracy']:.2f} | {average:.2f} | "
                     f"{summary['final_val']['loss']:.4f} | {summary['elapsed_seconds_this_invocation']:.2f} |")
        ignored = {"pruner", "output_dir", "results_file"}
        signatures.append(json.dumps({"args": {k: v for k, v in args.items() if k not in ignored},
                                      "data": data, "model": {k: v for k, v in model.items() if k != "pruner_name"}},
                                     sort_keys=True))
    lines.extend(["", "Accuracy is in percent. Total time includes loading, training, validation and saving; it is not inference latency.",
                  "Last-10 mean describes epochs within one run, not independent repetitions."])
    if len(set(signatures)) > 1:
        lines.append("WARNING: configurations differ beyond pruner/output paths; inspect run details before comparing.")
    if len(set(names)) != len(names):
        lines.append("WARNING: repeated pruners in the selected tail; these are not necessarily four distinct methods.")
    lines.extend(["", "Run details:"])
    for record in records:
        config = record["configuration"]
        args, data, model = config["args"], config["data"], config["model"]
        lines.append(f"- {args['pruner']} / {record['run_id']}: seed={args['seed']}, "
                     f"classes={len(data['class_names'])}, train/val={data['train_samples']}/{data['val_samples']}, "
                     f"image={model['image_size']}, dim={model['embed_dim']}, epochs={args['epochs']}; "
                     f"output={args['output_dir']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-file", type=Path,
                        default=Path(__file__).resolve().parent / "checkpoints" / "training_results.jsonl")
    parser.add_argument("--last", type=int, default=4)
    parser.add_argument("--output", type=Path, help="optionally save the table as Markdown")
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() == args.results_file.resolve():
            raise ValueError("Output must differ from the input JSONL file")
        records = load_tail(args.results_file, args.last)
        report = render_table(records)
        if len(records) < args.last:
            report = f"Only {len(records)} records available (requested {args.last}).\n\n" + report
        print(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
