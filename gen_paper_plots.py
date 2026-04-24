#!/usr/bin/env python3

import os

from cluster_cost_plot import cluster_cost_plot

OUTPUT_DIR = "plots/paper"
RESULT_DIR = "plots/paper"

plots = [
    {   
        "title"   : f"Cluster cost on bank (protected gender)",
        "data"    : f"{RESULT_DIR}/no_anchor/results_bank_normalized_cut_2026-04-24_00-11-42.csv",
        "plotfile": f"{OUTPUT_DIR}/cluster_cost_bank_no_anchor.pdf",
        "k"       : [1,2,5,10,20,30,40,50,100],
        "no_legend": False
    },
    {   
        "title"   : f"Cluster cost on income (protected gender)",
        "data"    : f"{RESULT_DIR}/no_anchor/results_income_normalized_cut_2026-04-24_01-10-23.csv",
        "plotfile": f"{OUTPUT_DIR}/cluster_cost_income_no_anchor.pdf",
        "k"       : [1,2,5,10,20,30,40,50,100],
        "no_legend": True
    },
    {   
        "title"   : f"Cluster cost on diabetes (protected gender)",
        "data"    : f"{RESULT_DIR}/no_anchor/results_diabetes_normalized_cut_2026-04-24_00-30-03.csv",
        "plotfile": f"{OUTPUT_DIR}/cluster_cost_diabetes_no_anchor.pdf",
        "k"       : [1,2,5,10,20,30,40,50,100],
        "no_legend": True
    },
    {   
        "title"   : f"Cluster cost on census (protected gender)",
        "data"    : f"{RESULT_DIR}/no_anchor/results_census_normalized_cut_2026-04-24_00-23-44.csv",
        "plotfile": f"{OUTPUT_DIR}/cluster_cost_census_no_anchor.pdf",
        "k"       : [1,2,5,10,20,30,40,50,100],
        "no_legend": True
    },
    {   
        "title"   : f"Cluster cost on diabetes (protected race)",
        "data"    : f"{RESULT_DIR}/no_anchor/results_diabetesrace_normalized_cut_2026-04-24_00-06-16.csv",
        "plotfile": f"{OUTPUT_DIR}/cluster_cost_diabetes_race_no_anchor.pdf",
        "k"       : [1,2,3,4,5,6,7,8,9,10],
        "no_legend": True
    },
    {   
        "title"   : f"Cluster cost on census (protected race)",
        "data"    : f"{RESULT_DIR}/no_anchor/results_censusrace_normalized_cut_2026-04-24_00-05-47.csv",
        "plotfile": f"{OUTPUT_DIR}/cluster_cost_census_race_no_anchor.pdf",
        "k"       : [1,2,3,4,5,6,7,8,9,10],
        "no_legend": True
    },
    {   
        "title"   : f"Cluster cost on bank (protected gender)",
        "data"    : f"{RESULT_DIR}/anchor/results_bank_normalized_cut_2026-04-23_23-04-36.csv",
        "plotfile": f"{OUTPUT_DIR}/cluster_cost_bank_anchor.pdf",
        "k"       : [1,2,5,10,20,30,40,50,100],
        "no_legend": False
    },
    {   
        "title"   : f"Cluster cost on income (protected gender)",
        "data"    : f"{RESULT_DIR}/anchor/results_income_normalized_cut_2026-04-23_19-36-34.csv",
        "plotfile": f"{OUTPUT_DIR}/cluster_cost_income_anchor.pdf",
        "k"       : [1,2,5,10,20,30,40,50,100],
        "no_legend": True
    },
    {   
        "title"   : f"Cluster cost on diabetes (protected gender)",
        "data"    : f"{RESULT_DIR}/anchor/results_diabetes_normalized_cut_2026-04-23_17-03-42.csv",
        "plotfile": f"{OUTPUT_DIR}/cluster_cost_diabetes_anchor.pdf",
        "k"       : [1,2,5,10,20,30,40,50,100],
        "no_legend": True
    },
    {   
        "title"   : f"Cluster cost on census (protected gender)",
        "data"    : f"{RESULT_DIR}/anchor/results_census_normalized_cut_2026-04-23_19-55-38.csv",
        "plotfile": f"{OUTPUT_DIR}/cluster_cost_census_anchor.pdf",
        "k"       : [1,2,5,10,20,30,40,50,100],
        "no_legend": True
    },
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for p in plots:
    cluster_cost_plot(p["data"], output=p["plotfile"], ks=p["k"], title=p["title"], no_legend=p["no_legend"])