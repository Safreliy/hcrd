#!/usr/bin/env python3
"""Generate compact manuscript tables and reports for the Q1 sensitivities."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _interval_tex(summary: dict) -> str:
    low, high = summary["percentile_95_ci"]
    return f"{_fmt(summary['mean'])} [{_fmt(low)}, {_fmt(high)}]"


def _bootstrap_tex(comparison: dict) -> str:
    low, high = comparison["bootstrap_95_ci"]
    return f"{_fmt(comparison['difference'])} [{_fmt(low)}, {_fmt(high)}]"


def _holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> None:
    refit_path = PROJECT / "results" / "ms_metrics_e2_refit_sensitivity" / "summary.json"
    files_path = PROJECT / "results" / "ms_metrics_e2_file_group_sensitivity" / "summary.json"
    qscore_path = PROJECT / "results" / "qscore_implementation_sensitivity" / "summary.json"
    missing = [path for path in (refit_path, files_path, qscore_path) if not path.exists()]
    if missing:
        raise SystemExit("missing sensitivity results: " + ", ".join(map(str, missing)))
    refit = _load(refit_path)
    files = _load(files_path)
    qscore = _load(qscore_path)

    directions = ("falkor_to_mesoscope", "mesoscope_to_falkor")
    labels = ("Falkor $\\to$ MESOSCOPE", "MESOSCOPE $\\to$ Falkor")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{E2 dependence and refitting sensitivity for HCRD-8+Q minus qscore AP. RT-block rows refit the source model and give bootstrap mean and percentile 95\% interval. The file row gives the range across ten delete-group representation-and-model refits.}",
        r"\label{tab:e2-refit-sensitivity}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        f"Design & {labels[0]} & {labels[1]} " + r"\\",
        r"\midrule",
    ]
    for block in (30, 60, 120):
        values = [
            refit["directions"][str(block)][direction]["hcrd_8_q_minus_qscore"]
            for direction in directions
        ]
        lines.append(
            f"Source-refit RT blocks ({block} s) & {_interval_tex(values[0])} & {_interval_tex(values[1])} "
            + r"\\"
        )
    file_values = [files["directions"][direction] for direction in directions]
    lines.extend(
        [
            "Acquisition-file delete-group & "
            + " & ".join(
                f"{_fmt(value['delta_min'])}--{_fmt(value['delta_max'])} ({value['positive_folds']}/{value['evaluated_folds']} positive)"
                for value in file_values
            )
            + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    generated = PROJECT / "paper" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "e2_refit_sensitivity_table.tex").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )

    q_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{qscore implementation sensitivity. Entries are HCRD-8 plus the indicated qscore summary minus that same qscore alone in target AP, with paired target-feature 95\% bootstrap intervals.}",
        r"\label{tab:qscore-sensitivity}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrcc}",
        r"\toprule",
        f"Variant & Width & {labels[0]} & {labels[1]} " + r"\\",
        r"\midrule",
    ]
    variant_labels = {
        "q_current": "Current median, $m\\geq8$",
        "q_min5": "Median, $m\\geq5$",
        "q_author5": "Author-like five-summary, $m\\geq5$",
        "q_multi7": "Multisummary seven-variable, $m\\geq8$",
    }
    for variant, label in variant_labels.items():
        entries = [qscore["directions"][direction][variant] for direction in directions]
        q_lines.append(
            f"{label} & {entries[0]['qscore_width']} & "
            f"{_bootstrap_tex(entries[0]['comparison'])} & "
            f"{_bootstrap_tex(entries[1]['comparison'])} "
            + r"\\"
        )
    q_lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    (generated / "qscore_sensitivity_table.tex").write_text(
        "\n".join(q_lines), encoding="utf-8", newline="\n"
    )

    report = [
        "# E2 source-refit and acquisition-file sensitivity",
        "",
        "The source learner was refit inside every RT-block bootstrap replicate; target RT blocks were resampled independently and paired across representations. The primary block width used 1,000 replicates, and the two width sensitivities used 300 each.",
        "",
        "| Design | Falkor to MESOSCOPE | MESOSCOPE to Falkor |",
        "|---|---:|---:|",
    ]
    for block in (30, 60, 120):
        values = [
            refit["directions"][str(block)][direction]["hcrd_8_q_minus_qscore"]
            for direction in directions
        ]
        report.append(
            f"| Source-refit RT blocks ({block} s) | {_interval_tex(values[0])} | {_interval_tex(values[1])} |"
        )
    report.append(
        "| Acquisition-file delete-group range | "
        + " | ".join(
            f"{_fmt(value['delta_min'])} to {_fmt(value['delta_max'])}; {value['positive_folds']}/{value['evaluated_folds']} positive"
            for value in file_values
        )
        + " |"
    )
    report.extend(
        [
            "",
            "The source-refit mean was positive in every direction and block-width design. The fraction of positive paired replicates was "
            + "; ".join(
                f"{block} s: {refit['directions'][str(block)][directions[0]]['hcrd_8_q_minus_qscore']['positive_fraction']:.1%}/{refit['directions'][str(block)][directions[1]]['hcrd_8_q_minus_qscore']['positive_fraction']:.1%}"
                for block in (30, 60, 120)
            )
            + " for Falkor-to-MESOSCOPE/MESOSCOPE-to-Falkor. All six percentile intervals nevertheless crossed zero, so this sensitivity supports a positive point effect but not a bootstrap sign claim after source refitting and RT-block resampling.",
            "",
            "The RT-block intervals propagate source refitting and local retention-time dependence. The deterministic file deletion repeats per-file aggregation and model fitting. Neither analysis can recover unavailable compound/adduct identifiers or represent additional laboratories.",
            "",
        ]
    )
    (PROJECT / "reports" / "ms_metrics_e2_refit_sensitivity.md").write_text(
        "\n".join(report), encoding="utf-8", newline="\n"
    )

    q_report = [
        "# qscore implementation sensitivity",
        "",
        "Every row compares HCRD-8 plus a qscore variant with that same qscore variant alone under the unchanged two-direction transfer learner.",
        "",
        "| Variant | Width | Falkor to MESOSCOPE | MESOSCOPE to Falkor |",
        "|---|---:|---:|---:|",
    ]
    for variant, label in variant_labels.items():
        entries = [qscore["directions"][direction][variant] for direction in directions]
        q_report.append(
            f"| {label.replace('$', '').replace('\\\\geq', '>=')} | {entries[0]['qscore_width']} | "
            f"{_bootstrap_tex(entries[0]['comparison'])} | "
            f"{_bootstrap_tex(entries[1]['comparison'])} |"
        )
    raw_p = [
        qscore["directions"][direction][variant]["comparison"]["two_sided_bootstrap_p"]
        for direction in directions
        for variant in variant_labels
    ]
    holm = _holm(raw_p)
    q_report.extend(
        [
            "",
            f"All eight intervals exclude zero; the largest Holm-adjusted bootstrap value is {max(holm):.6f}.",
            "",
            "## Falkor author-output fidelity",
            "",
            "```json",
            json.dumps(qscore["falkor_author_output_fidelity"], indent=2),
            "```",
            "",
            qscore["limitation"].capitalize().replace("mesoscope", "MESOSCOPE") + ".",
            "",
        ]
    )
    (PROJECT / "reports" / "qscore_implementation_sensitivity.md").write_text(
        "\n".join(q_report), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
