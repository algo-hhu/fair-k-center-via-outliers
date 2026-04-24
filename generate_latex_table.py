import ast
import re
from pathlib import Path

import pandas as pd


def generate_true_ratios_table(log_dir: str | Path = "logs/cluster_cost") -> str:
    log_dir = Path(log_dir)
    csvfiles = sorted(log_dir.glob("results_*.csv"))

    frames = []
    for f in csvfiles:
        df = pd.read_csv(f, sep=";", index_col=False)
        if df.empty or "true_ratios" not in df.columns:
            continue
        frames.append(df[["dataset", "subsample", "true_ratios", "ideal_ratios"]])

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["dataset", "subsample"])

    def parse_ratios(s):
        return ast.literal_eval(s)

    def instance_id(path):
        m = re.search(r"_(\d+)\.csv$", path)
        return int(m.group(1)) if m else -1

    df["instance"]            = df["subsample"].apply(instance_id)
    df["ratios"]              = df["true_ratios"].apply(parse_ratios)
    df["ideal_ratios_parsed"] = df["ideal_ratios"].apply(parse_ratios)
    df = df.sort_values(["dataset", "instance"])

    n_groups = df["ratios"].apply(len).max()
    group_cols = " & ".join(f"G{i+1}" for i in range(n_groups))
    header = r"Inst. & " + group_cols + r" & Ratio & Ideal \\"
    col_spec = "l" + "r" * n_groups + "rr"

    def build_table(dataset_df, dataset_name):
        lines = [
            f"% {dataset_name}",
            f"\\twocolumn[{{\\textbf{{{dataset_name}}}\\\\[4pt]}}]",
            r"{\small",
            r"\begin{longtblr}[",
            r"]{",
            f"  colspec = {{{col_spec}}},",
            r"  rowhead = 1,",
            r"}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        for _, row in dataset_df.iterrows():
            ratios = row["ratios"]
            min_r = min(ratios)
            ratio_cells = " & ".join(f"\\textbf{{{r}}}" if r == min_r else str(r) for r in ratios)
            q = max(ratios) / min(ratios)
            ratio_str = f"1:{q:.2f}" if ratios[0] <= ratios[1] else f"{q:.2f}:1"
            ideal_str = ":".join(str(r) for r in row["ideal_ratios_parsed"])
            lines.append(f"{row['instance']:05d} & {ratio_cells} & {ratio_str} & {ideal_str} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{longtblr}}")
        lines.append(r"\onecolumn")
        return "\n".join(lines)

    tables = []
    for ds, group in df.groupby("dataset", sort=True):
        tables.append(build_table(group, ds))

    return "\n\n".join(tables)


if __name__ == "__main__":
    table = generate_true_ratios_table()
    output = Path("true_ratios_table.txt")
    output.write_text(table)
    print(f"Table written to {output}")
