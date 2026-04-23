import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_cli():
    parser = argparse.ArgumentParser(description="Plot cluster cost results from a CSV file.")
    parser.add_argument("csvfile", type=str, help="Path to the CSV results file.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file for the plot (default: show interactively).")
    return parser.parse_args()

def cluster_cost_plot(csvfile, output: None):
    df = pd.read_csv(csvfile, sep=";", index_col=False, usecols=["k", "our_cost", "ref_cost", "our_fairlet_cost", "ref_fairlet_cost"])

    grouped = df.groupby("k")
    means = grouped.mean().sort_index()
    mins  = grouped.min().sort_index()
    maxs  = grouped.max().sort_index()

    plt.figure(figsize=(10, 6))
    our_line, = plt.plot(means.index, means["our_cost"], marker="o", label="our cost (mean)", linewidth=2, markersize=8)
    plt.fill_between(means.index, mins["our_cost"], maxs["our_cost"], alpha=0.2, color=our_line.get_color(), label="our cost (min–max)")
    ref_line, = plt.plot(means.index, means["ref_cost"], marker="s", label="ref cost (mean)", linewidth=2, markersize=8)
    plt.fill_between(means.index, mins["ref_cost"], maxs["ref_cost"], alpha=0.2, color=ref_line.get_color(), label="ref cost (min–max)")
    plt.plot(means.index, means["our_fairlet_cost"],  marker="^", label="our fairlet cost",  linewidth=2, markersize=8)
    plt.plot(means.index, means["ref_fairlet_cost"],  marker="D", label="ref fairlet cost",  linewidth=2, markersize=8)

    plt.xlabel("Number of clusters (k)", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    plt.title("Cluster and fairlet costs", fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(ticks=means.index, labels=[str(k) for k in means.index], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output}")
    else:
        plt.show()


def main():
    args = parse_cli()
    cluster_cost_plot(args.csvfile, args.output)

if __name__ == "__main__":
    main()
