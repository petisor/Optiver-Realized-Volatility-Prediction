"""
make_plots.py -- M5 writeup figures for the Optiver realized-volatility project.

Regenerates the three README figures straight from the raw order-book data,
using the SAME feature engineering and tuned-model settings as
explore_data.ipynb, on the small [0, 1, 2, 3] stock subset so it runs in
seconds (see CLAUDE.md's "small subset" rule).

Run from the project root:
    .venv/bin/python make_plots.py

Outputs (created in plots/):
    plots/volatility_clustering.png
    plots/baseline_vs_model.png
    plots/feature_correlation.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Shared look: recessive axes/grid, one colorblind-safe data hue + a neutral
# gray for reference lines. Keeping to a single data colour means we never
# rely on colour alone to tell things apart -- every value is labelled too.
# ----------------------------------------------------------------------------
BLUE = "#4C78A8"   # single data hue (colorblind-safe blue)
GRAY = "#8C8C8C"   # neutral: reference lines, the y = x diagonal

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.spines.top": False,     # drop the top/right box lines -> less clutter
    "axes.spines.right": False,
    "axes.edgecolor": "#666666",
    "axes.grid": True,
    "grid.color": "#E6E6E6",      # faint grid sits behind the data
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

SUBSET_STOCK_IDS = [0, 1, 2, 3]
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "realized_vol",
    "realized_vol_last_300",
    "spread_mean",
    "size_imbalance_mean",
    "imbalance_intensity",
]


def rmspe(y_true, y_pred):
    """Root Mean Squared Percentage Error -- the competition's metric."""
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


def compute_features_for_stock(stock_id):
    """One row of features per (stock_id, time_id) window -- same logic as the notebook."""
    book = pd.read_parquet(f"data/book_train.parquet/stock_id={stock_id}")

    # Weighted average price: the fair mid-price, weighting each side by the
    # OTHER side's size (a big resting order pulls the fair price toward it).
    book["wap"] = (
        (book["bid_price1"] * book["ask_size1"] + book["ask_price1"] * book["bid_size1"])
        / (book["bid_size1"] + book["ask_size1"])
    )
    # Log return between consecutive snapshots, WITHIN each window only.
    book["log_return"] = np.log(book["wap"] / book.groupby("time_id")["wap"].shift(1))
    book["spread"] = book["ask_price1"] - book["bid_price1"]
    book["size_imbalance"] = book["bid_size1"] / (book["bid_size1"] + book["ask_size1"])

    # Volatility computed only over the last 300s of the window (closest in time
    # to the window we are trying to predict).
    last_300 = book[book["seconds_in_bucket"] >= 300]
    realized_vol_last_300 = (
        last_300.groupby("time_id")["log_return"].apply(lambda x: np.sqrt((x ** 2).sum()))
    )

    grouped = book.groupby("time_id")
    features = pd.DataFrame({
        "realized_vol": grouped["log_return"].apply(lambda x: np.sqrt((x ** 2).sum())),
        "realized_vol_last_300": realized_vol_last_300,
        "spread_mean": grouped["spread"].mean(),
        "size_imbalance_mean": grouped["size_imbalance"].mean(),
    }).reset_index()

    features["realized_vol_last_300"] = features["realized_vol_last_300"].fillna(
        features["realized_vol"]
    )
    features["stock_id"] = stock_id
    return features


def build_feature_table():
    """Features for the subset, merged with the true target from train.csv."""
    train = pd.read_csv("data/train.csv")
    features = pd.concat(
        [compute_features_for_stock(s) for s in SUBSET_STOCK_IDS],
        ignore_index=True,
    )
    features = features.merge(train, on=["stock_id", "time_id"])
    # V-shape feature: distance of the imbalance from a balanced 0.5.
    features["imbalance_intensity"] = (features["size_imbalance_mean"] - 0.5).abs()
    return features


def tuned_out_of_fold_predictions(features):
    """Out-of-fold predictions from the tuned LightGBM (params chosen in the notebook)."""
    X = features[FEATURE_COLS]
    y = features["target"]
    groups = features["time_id"]  # keep one market moment inside a single fold

    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(features))
    for train_idx, val_idx in gkf.split(X, y, groups):
        model = lgb.LGBMRegressor(
            n_estimators=126,       # median tree count chosen by early stopping
            learning_rate=0.0437,   # chosen by RandomizedSearchCV
            num_leaves=15,          # chosen by RandomizedSearchCV
            random_state=42, verbose=-1, n_jobs=1,
        )
        # weight = 1/target^2 so training optimizes percentage error (matches RMSPE)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            sample_weight=1.0 / np.square(y.iloc[train_idx]),
        )
        oof_pred[val_idx] = model.predict(X.iloc[val_idx])
    return oof_pred


# ----------------------------------------------------------------------------
# Figure 1 -- volatility clustering (why the naive baseline is strong)
# ----------------------------------------------------------------------------
def plot_volatility_clustering(features):
    stock_0 = features[features["stock_id"] == 0].sort_values("time_id").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(range(len(stock_0)), stock_0["realized_vol"], color=BLUE, linewidth=1.4)
    ax.set_title("Volatility comes in bursts, it does not jump around randomly (stock 0)")
    ax.set_xlabel("10-minute window (in order)")
    ax.set_ylabel("realized volatility")
    ax.margins(x=0.01)

    out = PLOTS_DIR / "volatility_clustering.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Figure 2 -- naive baseline vs tuned model (predicted vs actual)
# ----------------------------------------------------------------------------
def plot_baseline_vs_model(features, oof_pred):
    y = features["target"].values
    naive_pred = features["realized_vol"].values

    naive_rmspe = rmspe(y, naive_pred)
    model_rmspe = rmspe(y, oof_pred)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharex=True, sharey=True)
    panels = [
        (naive_pred, "Naive baseline", naive_rmspe),
        (oof_pred, "Tuned LightGBM", model_rmspe),
    ]
    upper = max(y.max(), naive_pred.max(), oof_pred.max()) * 1.02
    for ax, (pred, label, score) in zip(axes, panels):
        ax.scatter(y, pred, s=7, alpha=0.25, color=BLUE, edgecolors="none")
        ax.plot([0, upper], [0, upper], color=GRAY, linewidth=1.2, linestyle="--")
        ax.set_title(f"{label}\nRMSPE = {score:.3f}")
        ax.set_xlabel("actual next-window volatility")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_aspect("equal")
    axes[0].set_ylabel("predicted volatility")
    fig.suptitle("Predicted vs actual: points nearer the dashed y = x line are better",
                 y=1.02, fontsize=11)

    out = PLOTS_DIR / "baseline_vs_model.png"
    fig.savefig(out)
    plt.close(fig)
    return out, naive_rmspe, model_rmspe


# ----------------------------------------------------------------------------
# Figure 3 -- feature correlation with the target (what helped, what didn't)
# ----------------------------------------------------------------------------
def plot_feature_correlation(features):
    corr = (
        features[FEATURE_COLS + ["target"]].corr()["target"].drop("target").sort_values()
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.barh(corr.index, corr.values, color=BLUE, height=0.65)
    ax.axvline(0, color=GRAY, linewidth=1)
    for i, value in enumerate(corr.values):
        # Always label just to the RIGHT of the zero line so tiny negative bars
        # don't push their label left into the y-axis feature names.
        label_x = value + 0.015 if value >= 0 else 0.015
        ax.text(label_x, i, f"{value:.2f}", va="center", ha="left",
                fontsize=9, color="#333333")
    ax.set_title("How strongly each feature moves with the target")
    ax.set_xlabel("correlation with target  (-1 to +1)")
    ax.set_xlim(-0.1, 1.0)

    out = PLOTS_DIR / "feature_correlation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    print("Building feature table for stocks", SUBSET_STOCK_IDS, "...")
    features = build_feature_table()
    print(f"  feature table: {features.shape[0]} rows")

    print("Training tuned LightGBM (out-of-fold) ...")
    oof_pred = tuned_out_of_fold_predictions(features)

    f1 = plot_volatility_clustering(features)
    (f2, naive_rmspe, model_rmspe) = plot_baseline_vs_model(features, oof_pred)
    f3 = plot_feature_correlation(features)

    print("\nSaved figures:")
    for f in (f1, f2, f3):
        print("  ", f)
    print(f"\nNaive baseline RMSPE : {naive_rmspe:.5f}")
    print(f"Tuned LightGBM RMSPE : {model_rmspe:.5f}")
    print(f"Improvement          : {(1 - model_rmspe / naive_rmspe) * 100:.1f}% lower error")


if __name__ == "__main__":
    main()
