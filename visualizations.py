# =============================================================
#  CSC-383 Data Visualization – Final Assignment
#  Student  : Muhammad Bilal
#  Roll No  : 22-DS-24
#  Session  : BSDS 4th Semester
#  Inst.    : Ms. Sana Younas
#  Dataset  : World Happiness Report (2018-2023)
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Output folder ────────────────────────────────────────────
CHARTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(CHARTS, exist_ok=True)

# ── Global Plot Style ────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "figure.dpi"        : 150,
    "savefig.bbox"      : "tight",
    "savefig.dpi"       : 150,
})

PALETTE  = ["#2C7BB6", "#1A9641", "#FDAE61", "#D7191C", "#7B2D8B", "#E66C2C"]
BG       = "#F8F9FA"
np.random.seed(42)

# =============================================================
#  PART 1 – DATASET CREATION
# =============================================================
print("=" * 55)
print("  CSC-383  |  Data Visualization  |  Muhammad Bilal")
print("=" * 55)
print("\n[1] Creating Dataset...")

REGIONS = {
    "Western Europe"    : ["Finland","Denmark","Iceland","Netherlands","Norway",
                            "Sweden","Switzerland","Luxembourg","Austria","Germany"],
    "North America"     : ["Canada","United States","Mexico","Costa Rica","Guatemala",
                            "Honduras","Panama","Jamaica","Dominican Rep.","Belize"],
    "Latin America"     : ["Uruguay","Chile","Brazil","Argentina","Colombia",
                            "Ecuador","Bolivia","Peru","Venezuela","Paraguay"],
    "Middle East"       : ["UAE","Qatar","Saudi Arabia","Kuwait","Bahrain",
                            "Israel","Jordan","Lebanon","Turkey","Iran"],
    "East Asia"         : ["Taiwan","Japan","South Korea","China","Singapore",
                            "Hong Kong","Malaysia","Thailand","Vietnam","Indonesia"],
    "Sub-Saharan Africa": ["Mauritius","Nigeria","Ghana","Kenya","Ethiopia",
                            "Rwanda","Senegal","Zambia","Zimbabwe","Sierra Leone"],
}

BASE = {
    "Western Europe"    : 74,
    "North America"     : 65,
    "Latin America"     : 60,
    "Middle East"       : 58,
    "East Asia"         : 62,
    "Sub-Saharan Africa": 42,
}

rows = []
for region, countries in REGIONS.items():
    b = BASE[region]
    for country in countries:
        hs  = float(np.clip(np.random.normal(b, 7), 28, 80)) / 10
        gdp = float(np.clip(np.random.normal(b * 0.022, 0.35), 0.30, 2.00))
        rows.append({
            "country"        : country,
            "region"         : region,
            "happiness_score": round(hs, 3),
            "gdp_per_capita" : round(gdp, 3),
            "social_support" : round(float(np.clip(np.random.normal(0.80, 0.12), 0.35, 0.99)), 3),
            "life_expectancy": round(float(np.clip(np.random.normal(b * 0.85, 4), 40.0, 78.0)), 1),
            "freedom"        : round(float(np.clip(np.random.normal(0.75, 0.13), 0.25, 0.98)), 3),
            "generosity"     : round(float(np.clip(np.random.normal(0.18, 0.12), 0.00, 0.70)), 3),
            "corruption"     : round(float(np.clip(np.random.normal(0.55, 0.18), 0.05, 0.95)), 3),
        })

df = pd.DataFrame(rows)

# ── Time-series (2018-2023) ──────────────────────────────────
YEARS = list(range(2018, 2024))
ts_rows = {}
for region, b in BASE.items():
    trend = np.linspace(b / 10, b / 10 + np.random.uniform(0.05, 0.30), len(YEARS))
    ts_rows[region] = np.clip(trend + np.random.normal(0, 0.07, len(YEARS)), 2.5, 8.0).round(3)
df_ts = pd.DataFrame(ts_rows, index=YEARS)

# ── Happiness tier column ────────────────────────────────────
df["tier"] = pd.cut(
    df["happiness_score"],
    bins   = [0, 4, 5, 6, 7, 10],
    labels = ["Very Low\n(<4)", "Low\n(4-5)", "Medium\n(5-6)", "High\n(6-7)", "Very High\n(>7)"],
)

# ── Print summary ────────────────────────────────────────────
print(f"   Rows    : {len(df)}")
print(f"   Columns : {df.shape[1]}")
print(f"   Regions : {df['region'].nunique()}")
print(f"   Years   : {YEARS[0]} – {YEARS[-1]}")
print("\n   Summary Statistics:")
print(df[["happiness_score","gdp_per_capita","life_expectancy",
          "freedom","corruption"]].describe().round(3).to_string())

# ── Save CSV ─────────────────────────────────────────────────
csv_path = os.path.join(CHARTS, "..", "happiness_dataset.csv")
df.to_csv(csv_path, index=False)
print(f"\n   Dataset saved → happiness_dataset.csv")

# =============================================================
#  PART 2 – EXPLORATORY DATA ANALYSIS
# =============================================================
print("\n[2] Part 2 – EDA Charts...")

NUM_COLS = ["happiness_score","gdp_per_capita","social_support",
            "life_expectancy","freedom","generosity","corruption"]

# ── 2a. Correlation Heatmap ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
corr = df[NUM_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 10, "weight": "bold"})
ax.set_title("Correlation Matrix – World Happiness Indicators",
             pad=15, fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "heatmap.png"))
plt.close()
print("   ✓ heatmap.png")

# ── 2b. Pair Plot ────────────────────────────────────────────
pp_df = df[["happiness_score","gdp_per_capita",
            "social_support","life_expectancy","freedom","region"]].copy()
g = sns.pairplot(pp_df, hue="region", palette="husl",
                 plot_kws={"alpha": 0.6, "s": 35},
                 diag_kind="kde", corner=True)
g.fig.suptitle("Pair Plot of Happiness Indicators",
               y=1.01, fontsize=14, fontweight="bold")
g.fig.set_size_inches(11, 9)
plt.savefig(os.path.join(CHARTS, "pairplot.png"))
plt.close()
print("   ✓ pairplot.png")

# =============================================================
#  PART 3 – CATEGORICAL VISUALIZATION
# =============================================================
print("\n[3] Part 3 – Categorical Charts...")

reg_mean = df.groupby("region", observed=True)["happiness_score"].mean().sort_values()

# ── 3a. Horizontal Bar Chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(reg_mean)))
bars = ax.barh(reg_mean.index, reg_mean.values,
               color=colors, height=0.6, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, reg_mean.values):
    ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=10, fontweight="bold", color="#333")
ax.axvline(reg_mean.mean(), color="#D7191C", linestyle="--",
           lw=1.8, alpha=0.8, label=f"Global Mean: {reg_mean.mean():.2f}")
ax.set_xlabel("Average Happiness Score (0–10)")
ax.set_title("Average Happiness Score by World Region", pad=12)
ax.set_xlim(0, 9)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "bar_chart.png"))
plt.close()
print("   ✓ bar_chart.png")

# ── 3b. Pie Chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
counts = df.groupby("region", observed=True)["happiness_score"].count()
wedges, texts, autotexts = ax.pie(
    counts.values, labels=counts.index, autopct="%1.0f%%",
    colors=PALETTE, explode=[0.04] * len(counts),
    startangle=140, pctdistance=0.78, labeldistance=1.08,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5})
for t in autotexts:
    t.set_fontsize(9); t.set_fontweight("bold")
ax.set_title("Country Distribution Across World Regions", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "pie_chart.png"))
plt.close()
print("   ✓ pie_chart.png")

# ── 3c. Count Plot ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
tier_counts = df["tier"].value_counts().sort_index()
tier_colors = ["#D7191C", "#FDAE61", "#FFFFBF", "#A6D96A", "#1A9641"]
bars2 = ax.bar(tier_counts.index.astype(str), tier_counts.values,
               color=tier_colors[:len(tier_counts)],
               width=0.55, edgecolor="white", linewidth=1)
for b in bars2:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
            str(int(b.get_height())), ha="center",
            fontsize=11, fontweight="bold")
ax.set_ylabel("Number of Countries")
ax.set_title("Count of Countries by Happiness Tier", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "count_plot.png"))
plt.close()
print("   ✓ count_plot.png")

# ── 3d. Stacked Bar Chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
stacked = df.groupby(["region", "tier"], observed=True).size().unstack(fill_value=0)
stacked.plot(kind="bar", stacked=True, ax=ax,
             color=tier_colors[:stacked.shape[1]],
             edgecolor="white", width=0.65)
ax.set_xlabel("Region"); ax.set_ylabel("Number of Countries")
ax.set_title("Happiness Tier Distribution by Region (Stacked)", pad=12)
ax.legend(title="Happiness Tier", bbox_to_anchor=(1.01, 1),
          loc="upper left", fontsize=9)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "stacked_bar.png"))
plt.close()
print("   ✓ stacked_bar.png")

# =============================================================
#  PART 4 – NUMERICAL VISUALIZATION
# =============================================================
print("\n[4] Part 4 – Numerical Charts...")

# ── 4a. Histograms ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)
for a in axes: a.set_facecolor(BG)

axes[0].hist(df["happiness_score"], bins=16, color="#2C7BB6",
             edgecolor="white", linewidth=0.8, alpha=0.9)
axes[0].axvline(df["happiness_score"].mean(), color="#D7191C", lw=2,
                linestyle="--", label=f"Mean: {df['happiness_score'].mean():.2f}")
axes[0].axvline(df["happiness_score"].median(), color="#FDAE61", lw=2,
                linestyle="-.", label=f"Median: {df['happiness_score'].median():.2f}")
axes[0].set_title("Distribution of Happiness Scores")
axes[0].set_xlabel("Happiness Score"); axes[0].set_ylabel("Frequency")
axes[0].legend()

axes[1].hist(df["gdp_per_capita"], bins=14, color="#1A9641",
             edgecolor="white", linewidth=0.8, alpha=0.9)
axes[1].axvline(df["gdp_per_capita"].mean(), color="#D7191C", lw=2,
                linestyle="--", label=f"Mean: {df['gdp_per_capita'].mean():.2f}")
axes[1].set_title("Distribution of GDP per Capita (log)")
axes[1].set_xlabel("GDP per Capita (log)"); axes[1].set_ylabel("Frequency")
axes[1].legend()

fig.suptitle("Histograms – Key Happiness Indicators",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "histogram.png"))
plt.close()
print("   ✓ histogram.png")

# ── 4b. Box Plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
region_names = list(REGIONS.keys())
data_by_region = [df[df["region"] == r]["happiness_score"].values
                  for r in region_names]
bp = ax.boxplot(data_by_region, patch_artist=True,
                medianprops={"color": "#D7191C", "linewidth": 2},
                whiskerprops={"linewidth": 1.5},
                capprops={"linewidth": 1.5},
                flierprops={"marker": "o", "markersize": 5, "alpha": 0.6})
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_xticks(range(1, len(region_names) + 1))
ax.set_xticklabels([r.replace(" ", "\n") for r in region_names], fontsize=9)
ax.set_ylabel("Happiness Score")
ax.set_title("Happiness Score Distribution by Region (Box Plot)", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "boxplot.png"))
plt.close()
print("   ✓ boxplot.png")

# ── 4c. Scatter Plot ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for i, region in enumerate(region_names):
    sub = df[df["region"] == region]
    ax.scatter(sub["gdp_per_capita"], sub["happiness_score"],
               color=PALETTE[i], label=region,
               s=60, alpha=0.85, edgecolors="white", linewidth=0.5)
# Regression line
slope, intercept, r_val, _, _ = stats.linregress(
    df["gdp_per_capita"], df["happiness_score"])
x_line = np.linspace(df["gdp_per_capita"].min(),
                     df["gdp_per_capita"].max(), 100)
ax.plot(x_line, slope * x_line + intercept,
        "k--", lw=2, alpha=0.7, label=f"Trend  r = {r_val:.2f}")
ax.set_xlabel("GDP per Capita (log scale)")
ax.set_ylabel("Happiness Score")
ax.set_title("GDP per Capita vs Happiness Score by Region", pad=12)
ax.legend(fontsize=8, ncol=2, framealpha=0.85)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "scatter.png"))
plt.close()
print("   ✓ scatter.png")

# ── 4d. Violin Plot ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
parts = ax.violinplot(data_by_region, showmedians=True, showextrema=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(PALETTE[i]); pc.set_alpha(0.70)
parts["cmedians"].set_color("#D7191C"); parts["cmedians"].set_linewidth(2.5)
ax.set_xticks(range(1, len(region_names) + 1))
ax.set_xticklabels([r.replace(" ", "\n") for r in region_names], fontsize=9)
ax.set_ylabel("Happiness Score")
ax.set_title("Violin Plot – Happiness Distribution by Region", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "violin.png"))
plt.close()
print("   ✓ violin.png")

# =============================================================
#  PART 5 – TIME-SERIES & TREND
# =============================================================
print("\n[5] Part 5 – Time-Series Charts...")

# ── 5a. Line Graph ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for i, region in enumerate(df_ts.columns):
    ax.plot(df_ts.index, df_ts[region], marker="o",
            linewidth=2.2, markersize=6,
            color=PALETTE[i], label=region)
ax.set_xlabel("Year"); ax.set_ylabel("Avg Happiness Score")
ax.set_title("Happiness Score Trends by Region (2018–2023)", pad=12)
ax.set_xticks(YEARS)
ax.legend(fontsize=9, framealpha=0.85)
ax.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "line_chart.png"))
plt.close()
print("   ✓ line_chart.png")

# ── 5b. Area Chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for i, region in enumerate(df_ts.columns):
    ax.fill_between(df_ts.index, df_ts[region],
                    alpha=0.22, color=PALETTE[i])
    ax.plot(df_ts.index, df_ts[region],
            linewidth=2, color=PALETTE[i], label=region)
ax.set_xlabel("Year"); ax.set_ylabel("Avg Happiness Score")
ax.set_title("Area Chart – Happiness Score Over Time by Region", pad=12)
ax.set_xticks(YEARS)
ax.legend(fontsize=9, framealpha=0.85)
ax.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "area_chart.png"))
plt.close()
print("   ✓ area_chart.png")

# ── 5c. Moving Average + Seasonal ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
for a in axes: a.set_facecolor(BG)

we_vals = df_ts["Western Europe"].values
ma = pd.Series(we_vals).rolling(2, min_periods=1).mean().values
axes[0].plot(YEARS, we_vals, "o-", color="#2C7BB6", lw=2, label="Actual")
axes[0].plot(YEARS, ma, "r--", lw=2.2, label="2-yr Moving Avg")
axes[0].set_title("Western Europe – Moving Average Trend")
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Happiness Score")
axes[0].legend(); axes[0].grid(alpha=0.3, linestyle="--")
axes[0].set_xticks(YEARS)

months = np.arange(1, 25)
seasonal = 7.2 + 0.018 * months + 0.11 * np.sin(2 * np.pi * months / 12)
noise    = np.random.normal(0, 0.045, len(months))
axes[1].plot(months, seasonal + noise, color="#1A9641", lw=1.6, label="Monthly Score")
axes[1].plot(months, seasonal, "k--", lw=2, label="Seasonal Trend")
axes[1].set_title("Seasonal Pattern – Western Europe (2022–2023)")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Happiness Score")
axes[1].legend(); axes[1].grid(alpha=0.3, linestyle="--")

fig.suptitle("Moving Average & Seasonal Analysis",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "moving_avg.png"))
plt.close()
print("   ✓ moving_avg.png")

# =============================================================
#  PART 6 – ADVANCED VISUALIZATIONS
# =============================================================
print("\n[6] Part 6 – Advanced Charts...")

# ── 6a. Bubble Chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for i, region in enumerate(region_names):
    sub = df[df["region"] == region]
    ax.scatter(sub["gdp_per_capita"], sub["happiness_score"],
               s=sub["social_support"] * 700,
               color=PALETTE[i], alpha=0.62,
               edgecolors="white", linewidth=0.8, label=region)
ax.set_xlabel("GDP per Capita (log)")
ax.set_ylabel("Happiness Score")
ax.set_title("Bubble Chart: GDP vs Happiness\n"
             "(Bubble Size = Social Support)", pad=12)
handles = [mpatches.Patch(color=PALETTE[i], label=r)
           for i, r in enumerate(region_names)]
ax.legend(handles=handles, fontsize=9, framealpha=0.85)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "bubble_chart.png"))
plt.close()
print("   ✓ bubble_chart.png")

# ── 6b. Tree Map ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
reg_vals = df.groupby("region", observed=True)["happiness_score"]\
             .mean().sort_values(ascending=False)
total = reg_vals.sum()
x = 0.0
for i, (region, val) in enumerate(reg_vals.items()):
    w = (val / total) * 10
    rect = plt.Rectangle((x, 0), w, 5,
                          color=PALETTE[i % len(PALETTE)],
                          alpha=0.88)
    ax.add_patch(rect)
    label = region.replace(" ", "\n")
    ax.text(x + w / 2, 2.5, f"{label}\n{val:.2f}",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    x += w
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
ax.set_title("Tree Map – Mean Happiness Score by Region",
             fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "treemap.png"))
plt.close()
print("   ✓ treemap.png")

# ── 6c. Executive Dashboard ──────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("#1E2B3A")
gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.42, wspace=0.35)

kpis = [
    ("Global Mean", f"{df['happiness_score'].mean():.2f}"),
    ("Happiest Region", "W. Europe"),
    ("Countries", str(len(df))),
]
for i, (label, val) in enumerate(kpis):
    ax_k = fig.add_subplot(gs[0, i])
    ax_k.set_facecolor("#28394A")
    ax_k.text(0.5, 0.58, val, ha="center", va="center",
              fontsize=20, fontweight="bold",
              color="#00D4AA", transform=ax_k.transAxes)
    ax_k.text(0.5, 0.22, label, ha="center", va="center",
              fontsize=10, color="#A0B4C4",
              transform=ax_k.transAxes)
    ax_k.set_xticks([]); ax_k.set_yticks([])
    for sp in ax_k.spines.values():
        sp.set_edgecolor("#3A5068"); sp.set_linewidth(1.5)

ax_l = fig.add_subplot(gs[1, :2])
ax_l.set_facecolor("#28394A")
for i, col in enumerate(list(df_ts.columns)[:4]):
    ax_l.plot(df_ts.index, df_ts[col],
              marker="o", ms=4, lw=1.8,
              color=PALETTE[i], label=col)
ax_l.set_title("Happiness Trends 2018–2023",
               color="white", fontweight="bold")
ax_l.tick_params(colors="white")
for sp in ax_l.spines.values(): sp.set_edgecolor("#3A5068")
ax_l.legend(fontsize=7, framealpha=0.3, labelcolor="white")
ax_l.grid(alpha=0.2, linestyle="--", color="white")
ax_l.set_xticks(YEARS)
ax_l.tick_params(axis="x", labelsize=8, colors="white")
ax_l.tick_params(axis="y", colors="white")

ax_b = fig.add_subplot(gs[1, 2])
ax_b.set_facecolor("#28394A")
top5 = df.nlargest(5, "happiness_score")[["country", "happiness_score"]]
ax_b.barh(top5["country"], top5["happiness_score"],
          color=["#00D4AA","#2C7BB6","#FDAE61","#A6D96A","#E66C2C"])
ax_b.set_title("Top 5 Countries", color="white", fontweight="bold")
ax_b.tick_params(colors="white")
ax_b.set_xlim(5, 9)
for sp in ax_b.spines.values(): sp.set_edgecolor("#3A5068")
ax_b.grid(axis="x", alpha=0.2, linestyle="--", color="white")
ax_b.tick_params(axis="both", colors="white", labelsize=8)

fig.suptitle("World Happiness Dashboard  |  2018–2023",
             fontsize=16, fontweight="bold",
             color="white", y=0.97)
plt.savefig(os.path.join(CHARTS, "dashboard.png"))
plt.close()
print("   ✓ dashboard.png")

# =============================================================
#  FINAL SUMMARY
# =============================================================
chart_files = [f for f in os.listdir(CHARTS) if f.endswith(".png")]
print("\n" + "=" * 55)
print(f"  ✅  All {len(chart_files)} charts generated successfully!")
print(f"  📁  Saved in: {CHARTS}")
print("=" * 55)
print("\n  Charts List:")
for f in sorted(chart_files):
    size_kb = os.path.getsize(os.path.join(CHARTS, f)) // 1024
    print(f"    {f:<22} {size_kb:>5} KB")
print()