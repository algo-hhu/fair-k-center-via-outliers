from pathlib import Path

import pandas as pd


def load_results(log_dir: str | Path = "logs/cluster_cost") -> pd.DataFrame:
    log_dir = Path(log_dir)
    frames = []
    for f in sorted(log_dir.glob("results_*.csv")):
        df = pd.read_csv(f, sep=";", index_col=False)
        if not df.empty and "our_cost" in df.columns:
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def generate_cost_table(log_dir: str | Path = "logs/cluster_cost") -> str:
    df = load_results(log_dir)

    cost_cols = ["our_cost", "ref_cost", "our_fairlet_cost", "ref_fairlet_cost"]
    summary = (
        df.groupby(["dataset", "k"])[cost_cols]
        .mean()
        .reset_index()
        .sort_values(["dataset", "k"])
    )

    col_spec = "llrrrr"
    header1 = r"Dataset & $k$ & \multicolumn{2}{c}{cost} & \multicolumn{2}{c}{fairlet cost} \\"
    header2 = r" &  & ours & Chier & ours & Chier \\"

    lines = [
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header1,
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        header2,
        r"\midrule",
    ]

    prev_dataset = None
    for _, row in summary.iterrows():
        dataset = row["dataset"]
        if prev_dataset is not None and dataset != prev_dataset:
            lines.append(r"\midrule")
        def fmt(a, b):
            sa, sb = f"{a:.3f}", f"{b:.3f}"
            if a < b:
                return f"\\textbf{{{sa}}}", sb
            elif b < a:
                return sa, f"\\textbf{{{sb}}}"
            return sa, sb

        cost_our, cost_ref = fmt(row["our_cost"], row["ref_cost"])
        fairlet_our, fairlet_ref = fmt(row["our_fairlet_cost"], row["ref_fairlet_cost"])
        cells = [
            dataset if dataset != prev_dataset else "",
            str(int(row["k"])),
            cost_our,
            cost_ref,
            fairlet_our,
            fairlet_ref,
        ]
        lines.append(" & ".join(cells) + r" \\")
        prev_dataset = dataset

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    table = generate_cost_table()
    output = Path("cost_table.txt")
    output.write_text(table)
    print(table)
    print(f"\nTable written to {output}")
