import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_cli():
    parser = argparse.ArgumentParser(description="Plot cluster cost results from a CSV file.")
    parser.add_argument("csvfile", type=str, help="Path to the CSV results file.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file for the plot (default: show interactively).")
    parser.add_argument("--ks", type=int, nargs="+", default=None,
                        help="List of k values to include in the plot (default: all).")
    parser.add_argument("--title", type=str, default="Cluster and fairlet costs",
                        help="Title for the plot.")
    parser.add_argument("--no-legend", action="store_true",
                        help="Omit the legend from the plot.")
    return parser.parse_args()

def cluster_cost_plot(csvfile, output=None, ks=None, title="Cluster and fairlet costs", no_legend=False):
    df = pd.read_csv(csvfile, sep=";", index_col=False, usecols=["k", "our_cost", "ref_cost", "our_fairlet_cost", "ref_fairlet_cost"])

    if ks is not None:
        df = df[df["k"].isin(ks)]

    grouped = df.groupby("k")
    means = grouped.mean().sort_index()
    mins  = grouped.min().sort_index()
    maxs  = grouped.max().sort_index()

    # Use integer positions so markers are equidistant on the x-axis
    x = list(range(len(means)))
    x_labels = [str(k) for k in means.index]

    _, ax = plt.subplots(figsize=(12, 8))
    our_line, = ax.plot(x, means["our_cost"], marker="o", label="Our cost (mean)", linewidth=2, markersize=8)
    ax.fill_between(x, mins["our_cost"], maxs["our_cost"], alpha=0.2, color=our_line.get_color(), label="Our cost (min–max)")
    ref_line, = ax.plot(x, means["ref_cost"], marker="s", label="Chier. cost (mean)", linewidth=2, markersize=8)
    ax.fill_between(x, mins["ref_cost"], maxs["ref_cost"], alpha=0.2, color=ref_line.get_color(), label="Chier. cost (min–max)")
    ax.plot(x, means["our_fairlet_cost"],  marker="^", label="Our fairlet cost",  linewidth=2, markersize=8)
    ax.plot(x, means["ref_fairlet_cost"],  marker="D", label="Chier. fairlet cost",  linewidth=2, markersize=8)

    ax.set_xlabel("Number of clusters (k)", fontsize=38)
    ax.set_ylabel("Cost", fontsize=38)
    ax.set_title(title, fontsize=40)
    ax.set_ylim(0.4, 3.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    if not no_legend:
        ax.legend(fontsize=25, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=30)
    ax.tick_params(axis="y", labelsize=30)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output}")
    else:
        plt.show()


def main():
    args = parse_cli()
    cluster_cost_plot(args.csvfile, args.output, args.ks, args.title, args.no_legend)

if __name__ == "__main__":
    main()
