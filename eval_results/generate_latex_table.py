#!/usr/bin/env python3
"""
Generate LaTeX table from eval_results/main_table.txt
Structure: Method | Response Level (macro P/R/F1, label=0 P/R/F1, label=1 P/R/F1) | Streaming Level (same)
"""

import re
from pathlib import Path


def parse_main_table(filepath: str) -> list[dict]:
    """Parse main_table.txt and return list of method results."""
    text = Path(filepath).read_text()

    # Method names mapping: Alpha=0 -> ResponseBCE, Alpha=1 -> TokenBCE, Alpha=0.5 -> BalancedBCE
    method_map = {
        "0": "ResponseBCE",
        "1": "TokenBCE",
        "0.5": "BalancedBCE",
    }

    # Find section starts: "Alpha = 0 ", "Alpha = 1 ", "Alpha = 0.5 " (avoid matching 0 in 0.5)
    section_starts = []
    for m in re.finditer(r"Alpha = (0\.5|0|1)\s", text):
        section_starts.append((m.start(), m.end(), m.group(1)))

    results = []
    for i, (start, header_end, alpha_val) in enumerate(section_starts):
        end = section_starts[i + 1][0] if i + 1 < len(section_starts) else len(text)
        block = text[header_end:end]

        resp_section = re.search(r"-------------Response level--------.*?weighted avg\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+", block, re.DOTALL)
        stream_section = re.search(r"-----------Streaming level-----------.*?weighted avg\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+", block, re.DOTALL)

        def extract_metrics(section_text):
            if not section_text:
                return None
            out = {}
            # macro avg: line like "   macro avg     0.5404    0.5417    0.5396        20"
            m = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", section_text)
            if m:
                out["macro"] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            # label 0: "           0     0.4444    0.5000    0.4706"
            m0 = re.search(r"\n\s+0\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", section_text)
            if m0:
                out["label0"] = (float(m0.group(1)), float(m0.group(2)), float(m0.group(3)))
            # label 1
            m1 = re.search(r"\n\s+1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", section_text)
            if m1:
                out["label1"] = (float(m1.group(1)), float(m1.group(2)), float(m1.group(3)))
            return out if out else None

        resp_metrics = extract_metrics(resp_section.group(0)) if resp_section else None
        stream_metrics = extract_metrics(stream_section.group(0)) if stream_section else None

        results.append({
            "method": method_map.get(alpha_val, f"Alpha={alpha_val}"),
            "response": resp_metrics,
            "streaming": stream_metrics,
        })

    return results


def fmt(x: float) -> str:
    return f"{x:.2f}"


def generate_latex(results: list[dict]) -> str:
    # Header: Method | Response Level (9 cols) | Streaming Level (9 cols)
    # Response/Streaming: macro (P,R,F1), label=0 (P,R,F1), label=1 (P,R,F1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc|ccc}",
        r"\toprule",
        r"& \multicolumn{9}{c|}{\textbf{Response Level}} & \multicolumn{9}{c}{\textbf{Streaming Level}} \\",
        r"\cmidrule(lr){2-10} \cmidrule(lr){11-19}",
        r"& \multicolumn{3}{c|}{macro avg} & \multicolumn{3}{c|}{label=0} & \multicolumn{3}{c|}{label=1} & \multicolumn{3}{c|}{macro avg} & \multicolumn{3}{c|}{label=0} & \multicolumn{3}{c}{label=1} \\",
        r"Method & P & R & F1 & P & R & F1 & P & R & F1 & P & R & F1 & P & R & F1 & P & R & F1 \\",
        r"\midrule",
    ]

    for r in results:
        resp = r["response"] or {}
        stream = r["streaming"] or {}
        row = [r["method"]]
        for section in [resp, stream]:
            for key in ["macro", "label0", "label1"]:
                t = section.get(key, (0, 0, 0))
                row.extend([fmt(t[0]), fmt(t[1]), fmt(t[2])])
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Evaluation results. P=precision, R=recall, F1=f1-score.}",
        r"\label{tab:eval-results}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    base = Path(__file__).resolve().parent
    input_path = base / "main_table.txt"
    output_path = base / "main_table.tex"

    results = parse_main_table(input_path)
    latex = generate_latex(results)
    output_path.write_text(latex, encoding="utf-8")
    print(f"Written: {output_path}")
    print(latex)


if __name__ == "__main__":
    main()
