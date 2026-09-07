"""CSV/Markdown and a standalone searchable HTML experiment dashboard."""
import csv
import html
import math
import statistics
from pathlib import Path

from experiment_data import read_json, atomic_json
from experiment_suite import load_project


def run_state(run):
    state = read_json(run / "state.json") if (run / "state.json").exists() else {"status": "pending"}
    if (run / "integrity.json").exists():
        integrity = read_json(run / "integrity.json")
        if integrity["status"] == "failed":
            state = {**state, "status": "integrity_error", "error": integrity["error"]}
    return state


def project_status(project):
    """Read atomic metadata without acquiring the training lock or loading weights."""
    project, _, _, plan = load_project(project)
    rows = []
    for job in plan["jobs"]:
        run = project / "runs" / job["run_id"]
        state = run_state(run)
        history = read_json(run / "metrics.json") if (run / "metrics.json").exists() else []
        rows.append(dict(run_id=job["run_id"], status=state["status"], epoch=history[-1]["epoch"] if history else 0,
                         dev_accuracy=history[-1]["val"]["accuracy"] if history else None))
    return rows


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def paired_summary(values):
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return dict(n=n, mean_delta_pp=mean, sd=None, ci_low=None, ci_high=None, conclusion="single-seed lead only")
    # Two-sided 95% Student-t quantiles. For larger n choose a conservative lower df.
    critical = {1: 12.706205, 2: 4.302653, 3: 3.182447, 4: 2.776445, 5: 2.570582,
                6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
                15: 2.131450, 20: 2.085964, 30: 2.042272, 60: 2.000298, 120: 1.979930}
    df = max(key for key in critical if key <= n - 1)
    sd = statistics.stdev(values)
    margin = critical[df] * sd / math.sqrt(n)
    low, high = mean - margin, mean + margin
    return dict(n=n, mean_delta_pp=mean, sd=sd, ci_low=low, ci_high=high,
                conclusion="supports improvement" if n >= 3 and mean >= .5 and low > 0 else "inconclusive/exploratory")


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else ["status"])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report_project(project, plots=False):
    project, settings, manifest, plan = load_project(project)
    output = project / "report"
    output.mkdir(exist_ok=True)
    rows, costs, analyses, finals, benchmarks = [], [], [], [], []
    for job in plan["jobs"]:
        run = project / "runs" / job["run_id"]
        state = run_state(run)
        history = read_json(run / "metrics.json") if (run / "metrics.json").exists() else []
        evaluation = read_json(run / "evaluation_dev.json") if (run / "evaluation_dev.json").exists() else {}
        row = dict(run_id=job["run_id"], model=job["model_id"], seed=job["seed"], status=state["status"],
            epoch=history[-1]["epoch"] if history else 0, dev_accuracy=evaluation.get("accuracy") if state["status"] == "complete" else
            (history[-1]["val"]["accuracy"] if history else None),
            dev_nll=evaluation.get("nll") if state["status"] == "complete" else None,
            train_seconds=sum(h.get("seconds", 0) for h in history), error=state.get("error", ""))
        rows.append(row)
        if (run / "cost.json").exists():
            costs.append(dict(run_id=job["run_id"], **read_json(run / "cost.json")))
        for path in sorted((run / "benchmarks").glob("*.json")):
            value = read_json(path)
            for batch in (1, 16):
                measured = [r for r in value["measurements"] if r["batch"] == batch]
                benchmarks.append(dict(run_id=job["run_id"], measurement=path.stem, batch=batch,
                    device=value["environment"]["device_name"], median_ms=statistics.median(r["median_ms"] for r in measured),
                    p90_ms=statistics.median(r["p90_ms"] for r in measured),
                    images_per_second=statistics.median(r["images_per_second"] for r in measured),
                    repeats=value["repeats"], iterations=value["iterations"],
                    peak_allocated_bytes=max((r["peak_allocated_bytes"] or 0 for r in measured)) or None))
        if (run / "evaluation_final.json").exists():
            value = read_json(run / "evaluation_final.json")
            finals.append(dict(run_id=job["run_id"], model=job["model_id"], seed=job["seed"],
                accuracy=value["accuracy"], nll=value["nll"], final_status=value["final_status"]))
        for report in sorted((run / "analysis").glob("*/report.json")):
            if "_interrupted_" not in report.parent.name:
                analyses.append((job["run_id"], report.parent.name, "../" + report.parent.relative_to(project).as_posix() + "/report.md"))
    completed = [r for r in rows if r["status"] == "complete" and r["dev_accuracy"] is not None]
    groups = {}
    for row in completed:
        groups.setdefault(row["model"], {})[row["seed"]] = row["dev_accuracy"]
    summaries = [dict(model=name, n=len(values), mean_accuracy=statistics.mean(values.values()),
                      sd=statistics.stdev(values.values()) if len(values) > 1 else None)
                 for name, values in sorted(groups.items())]
    pairs = []
    comparisons = {(name, "Q") for name in groups if name != "Q"}
    comparisons |= {("GQQ", "Q"), ("GGQ", "GQQ"), ("G", "GGQ"), ("G", "GQQ"), ("G", "E"),
                    ("G_centered", "G"), ("G_raw", "G"), ("Gplus", "G")}
    budget_rows = []
    for path in sorted(project.glob("budget_selection_*.json")):
        selection = read_json(path)
        for row in selection["comparisons"]:
            budget_rows.append({"metric": selection["metric"], **row})
            if row["matched"]:
                comparisons.add((row["target"], row["candidate"]))
    for a, b in sorted(comparisons):
        shared = sorted(set(groups.get(a, {})) & set(groups.get(b, {})))
        if shared:
            pairs.append(dict(model=a, reference=b, seeds=",".join(map(str, shared)),
                **paired_summary([groups[a][s] - groups[b][s] for s in shared])))
    write_csv(output / "runs.csv", rows)
    write_csv(output / "cost.csv", costs)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired.csv", pairs)
    write_csv(output / "final.csv", finals)
    write_csv(output / "benchmark.csv", benchmarks)
    write_csv(output / "budget_matches.csv", budget_rows)
    final_groups = {}
    for row in finals:
        final_groups.setdefault(row["model"], {})[row["seed"]] = row["accuracy"]
    final_pairs = []
    for a, b in sorted(comparisons):
        shared = sorted(set(final_groups.get(a, {})) & set(final_groups.get(b, {})))
        if shared:
            final_pairs.append(dict(model=a, reference=b, seeds=",".join(map(str, shared)),
                **paired_summary([final_groups[a][s] - final_groups[b][s] for s in shared])))
    write_csv(output / "final_paired.csv", final_pairs)
    atomic_json(output / "summary.json", dict(runs=rows, summary=summaries, paired=pairs, final=finals))
    lines = ["# TENet 实验总览", "", f"最终集状态：`{manifest['final_status']}`；当前主表使用 **dev / last checkpoint**。",
        "单种子结果仅供筛选；配对区间描述训练随机性，不等于总体保证。成本表的运算量只覆盖Conv/Linear。", "",
        "| Run | 状态 | 轮次 | Dev Top-1 (%) |", "| --- | --- | ---: | ---: |"]
    lines += [f"| {r['run_id']} | {r['status']} | {r['epoch']} | {r['dev_accuracy'] if r['dev_accuracy'] is not None else '—'} |" for r in rows]
    lines += ["", "详细结果：runs.csv、summary.csv、paired.csv、cost.csv、final.csv。", "",
              "## 同种子配对（探索性）", "", "| 模型 | 对照 | n | 差值 pp | 95%区间 | 判定 |", "| --- | --- | ---: | ---: | --- | --- |"]
    lines += [f"| {r['model']} | {r['reference']} | {r['n']} | {r['mean_delta_pp']:.3f} | {r['ci_low']}, {r['ci_high']} | {r['conclusion']} |" for r in pairs]
    if analyses:
        lines += ["", "## 分析入口", ""] + [f"- [{name} / {mode}]({link})" for name, mode, link in analyses]
    if finals:
        lines += ["", "## 冻结后的最终评估", "", "| Run | Top-1 (%) | 最终集状态 |", "| --- | ---: | --- |"]
        lines += [f"| {r['run_id']} | {r['accuracy']:.3f} | {r['final_status']} |" for r in finals]
    atomic_text(output / "report.md", "\n".join(lines) + "\n")
    def table(records, fields):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(f)}</th>" for f in fields) + "</tr></thead><tbody>" + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(r.get(f, '')))}</td>" for f in fields) + "</tr>" for r in records) + "</tbody></table>"
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TENet 实验总览</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;color:#172c40;background:#f5f8fb}table{border-collapse:collapse;width:100%;background:white;margin:20px 0;display:block;overflow:auto}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}input{padding:12px;width:90%;font:inherit}a{color:#1568a6}.note{padding:18px;background:#e3eef8}img{max-width:100%}</style>
<h1>TENet 实验总览</h1><p>训练每轮更新本页；运行中每30秒自动刷新。中间轮次不进入跨种子结论。可直接打开此 HTML，无需启动服务。</p>"""
    if any(row["status"] == "running" for row in rows):
        document += '<script>setTimeout(()=>location.reload(),30000)</script>'
    document += f"<p class='note'>Dev / last checkpoint；最终集：{html.escape(manifest['final_status'])}。单种子仅作线索；未运行项显示 pending。</p>"
    document += '<input id="search" placeholder="筛选模型、种子或状态" oninput="document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()))">'
    document += "<h2>运行进度</h2>" + table(rows, list(rows[0]) if rows else ["status"])
    document += "<h2>跨种子结果</h2>" + table(summaries, ["model", "n", "mean_accuracy", "sd"])
    document += "<h2>同种子配对（探索性）</h2>" + table(pairs, ["model", "reference", "n", "mean_delta_pp", "ci_low", "ci_high", "conclusion"])
    document += "<h2>成本（运算量仅覆盖 Conv/Linear）</h2>" + table(costs, ["run_id", "parameters", "head_parameters", "conv_linear_macs"])
    document += "<h2>预算匹配（未匹配不构成公平对照）</h2>" + table(budget_rows, ["metric", "target", "candidate", "relative_gap", "matched"])
    document += "<h2>硬件实测（不同环境单独比较）</h2>" + table(benchmarks, ["run_id", "device", "batch", "median_ms", "p90_ms", "images_per_second", "peak_allocated_bytes"])
    document += "<h2>冻结后的最终评估</h2>" + table(finals, ["run_id", "accuracy", "nll", "final_status"])
    document += '<h2>下载与分析</h2><p>' + " · ".join(f'<a href="{name}">{name}</a>' for name in ("report.md", "runs.csv", "paired.csv", "cost.csv", "benchmark.csv", "final.csv", "final_paired.csv")) + '</p><ul>'
    document += "".join(f'<li><a href="{html.escape(link, quote=True)}">{html.escape(name + " / " + mode)}</a></li>' for name, mode, link in analyses) + "</ul>"
    if plots and completed:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        cost_map = {r["run_id"]: r for r in costs}
        for row in completed:
            history = read_json(project / "runs" / row["run_id"] / "metrics.json")
            axes[0].plot([h["epoch"] for h in history], [h["val"]["accuracy"] for h in history], label=row["run_id"])
            if row["run_id"] in cost_map:
                axes[1].scatter(cost_map[row["run_id"]]["parameters"], row["dev_accuracy"])
                axes[1].annotate(row["run_id"], (cost_map[row["run_id"]]["parameters"], row["dev_accuracy"]), fontsize=7)
        axes[0].set(xlabel="Epoch", ylabel="Dev Top-1 (%)")
        axes[1].set(xlabel="Total parameters", ylabel="Dev Top-1 (%)")
        if len(completed) <= 8:
            axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "curves.png", dpi=160)
        fig.savefig(output / "curves.svg")
        plt.close(fig)
    if (output / "curves.png").exists():
        document += '<h2>训练曲线与参数对照</h2><img src="curves.png" alt="Training curves and parameter comparison">'
    atomic_text(output / "index.html", document + "</html>\n")
    return output / "index.html"
