"""Render manuscript figures solely from published aggregate tables."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/stage4/full_run"
OUT = ROOT / "reports/figures"
GROUPS = ["conservative18", "local93", "published165"]
TITLES = ["Conservative 18", "Published local 93", "Published non-time 165"]
MODELS = ["logistic_regression", "lightgbm", "graphsage", "gat", "sage_gru"]
LABELS = ["Logistic Regression", "LightGBM", "GraphSAGE", "GAT", "Historical GRU"]
COLORS = ["#777777", "#0072B2", "#E69F00", "#009E73", "#CC79A7"]
plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                     "svg.fonttype":"none"})
summary = pd.read_csv(SOURCE / "model_comparison.csv")
robust = pd.read_csv(SOURCE / "robustness_summary.csv")
intervals = pd.read_csv(SOURCE / "paired_bootstrap_intervals.csv")
periods = pd.read_csv(SOURCE / "per_time_step_metrics.csv")

def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(7,6.2), sharex=True, layout="constrained")
for ax,group,title in zip(axes,GROUPS,TITLES):
    rows = summary.loc[summary.feature_configuration.eq(group)].set_index("architecture")
    for y,(model,label,color) in enumerate(zip(MODELS,LABELS,COLORS)):
        row=rows.loc[model]
        ax.errorbar(row.ap_mean,y,xerr=None if pd.isna(row.ap_seed_sd) else row.ap_seed_sd,
                    fmt="o",color=color,capsize=3)
    ax.set(yticks=range(5),yticklabels=LABELS,title=title,xlim=(0,1))
    ax.invert_yaxis()
axes[-1].set_xlabel("Pooled test Average Precision")
save(fig,"editorial_pooled_ap")

rows=periods.loc[periods.model_id.eq("lightgbm__published165__ordinary__seed42") & periods.calibration.eq("raw")].sort_values("time_step")
assert len(rows)==15
assert rows.loc[rows.time_step.eq(46),"recall"].isna().all()
fig,ax=plt.subplots(figsize=(7,3.2),layout="constrained")
ax.plot(rows.time_step,rows.recall,marker="o",color=COLORS[1])
ax.axvline(42.5,color="#777777",ls="--",lw=1)
ax.set(ylim=(-.03,1.04),xticks=rows.time_step,xlabel="Time step\nKnown illicit labels",ylabel="Recall at frozen F1 threshold")
ax.set_xticklabels([f"{t}\n{n}" for t,n in zip(rows.time_step,rows.n_illicit)])
ax.text(46,.3,"Unavailable\n2 illicit labels",ha="center",fontsize=9)
save(fig,"editorial_recall")

fig,axes=plt.subplots(3,1,figsize=(7,6.5),sharex=True,sharey=True,layout="constrained")
for ax,group,title in zip(axes,GROUPS,TITLES):
    for model,label,color in zip(MODELS,LABELS,COLORS):
        view=robust.loc[robust.architecture.eq(model)&robust.feature_configuration.eq(group)&robust.perturbation.eq("noise")].sort_values("severity")
        base=summary.loc[summary.architecture.eq(model)&summary.feature_configuration.eq(group),"ap_mean"].item()
        ax.plot([0]+view.severity.tolist(),[base]+view.ap_mean.tolist(),marker="o",color=color,label=label)
    ax.set(title=title,ylim=(0,1),ylabel="Pooled AP",xticks=[0,.1,.2,.3])
axes[0].legend(ncol=3,fontsize=8,loc="upper right")
axes[-1].set_xlabel("Synthetic Gaussian noise scale / training SD")
save(fig,"editorial_noise")

fig,axes=plt.subplots(3,1,figsize=(7,5.6),sharex=True,layout="constrained")
names=dict(zip(MODELS,LABELS))
for ax,group,title in zip(axes,GROUPS,TITLES):
    view=intervals.loc[intervals.feature_configuration.eq(group)].reset_index(drop=True)
    for y,row in view.iterrows():
        ax.plot([row.ci_low,row.ci_high],[y,y],color=COLORS[1],lw=2)
        ax.scatter(row.observed_ap_difference,y,color="#D55E00",s=23)
    ax.axvline(0,color="#777777",ls=":")
    ax.set(title=title,yticks=range(4),yticklabels=[f"{names[r.first_architecture]} − {names[r.second_architecture]}" for r in view.itertuples()],xlim=(-.05,.6))
    ax.invert_yaxis()
axes[-1].set_xlabel("Pooled AP difference with descriptive 95% interval")
save(fig,"editorial_intervals")
