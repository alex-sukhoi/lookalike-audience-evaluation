"""
Lookalike audience evaluation
=============================

Question from the advertiser side: *how similar is a lookalike audience to my
customers, which model builds the best one, and how much quality do I lose when
I make the audience bigger?*

The difficulty: publisher users are unlabelled. We never observe true
negatives, so accuracy or ROC on the inference data cannot be computed.

Approach used here
------------------
1. **Seed hold-out recall** - hide 20% of the advertiser's customers, mix them
   into the publisher's inference pool, and measure what share of them each
   audience cut (top 1%, 2%, 5%, ...) recovers. A good model ranks hidden
   customers near the top. Recall@k divided by k is the *lift* over random.
2. **Similarity profile** - mean cosine similarity of each audience cut to the
   seed centroid, an interpretable "how alike are they" number that can be
   reported for every cut without labels.
3. **Ground truth check** - because the data are synthetic, we also know which
   publisher users are truly similar and can confirm that hold-out recall ranks
   the models in the right order.

Data are synthetic: 32-dimensional user embeddings for 60,000 publisher users
and 2,000 advertiser customers.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

RNG = np.random.default_rng(2024)
ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Synthetic data
# -----------------------------------------------------------------------------
DIM, N_PUB, N_SEED, N_INTEREST = 32, 60_000, 2_000, 12
centres = normalize(RNG.normal(size=(N_INTEREST, DIM)))          # interest clusters
target_mix = np.array([0.45, 0.30, 0.15] + [0.10 / 9] * 9)      # advertiser's customer profile
general_mix = np.full(N_INTEREST, 1 / N_INTEREST)


def users(n, mix, noise=0.9):
    """Users are a mixture of interest clusters plus individual noise."""
    k = RNG.choice(N_INTEREST, size=n, p=mix)
    emb = centres[k] * 1.6 + RNG.normal(scale=noise, size=(n, DIM))
    return emb.astype(np.float32), k


seed_X, _ = users(N_SEED, target_mix)
pub_X, pub_k = users(N_PUB, general_mix)
# Ground truth (unknown in practice): publisher users from the advertiser's top-2 interests
pub_truly_similar = np.isin(pub_k, [0, 1])

# Hold out 20% of seed customers and hide them in the inference pool
idx = RNG.permutation(N_SEED)
hold, train_seed = idx[:400], idx[400:]
infer_X = np.vstack([pub_X, seed_X[hold]])
is_holdout = np.r_[np.zeros(N_PUB, bool), np.ones(len(hold), bool)]
truly_similar = np.r_[pub_truly_similar, np.ones(len(hold), bool)]

# Training set: seed (label 1) vs a random sample of publisher users (label 0,
# "unlabelled treated as negative"), balanced as in production
neg_idx = RNG.choice(N_PUB, size=len(train_seed) * 3, replace=False)
X_train = np.vstack([seed_X[train_seed], pub_X[neg_idx]])
y_train = np.r_[np.ones(len(train_seed)), np.zeros(len(neg_idx))]

# -----------------------------------------------------------------------------
# 2. Models
# -----------------------------------------------------------------------------
seed_centroid = normalize(seed_X[train_seed].mean(axis=0, keepdims=True))
infer_unit = normalize(infer_X)
scores = {
    "Cosine to seed centroid": (infer_unit @ seed_centroid.T).ravel(),
    "Logistic regression": LogisticRegression(max_iter=2000, C=1.0)
        .fit(X_train, y_train).predict_proba(infer_X)[:, 1],
    "Gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                        random_state=0)
        .fit(X_train, y_train).predict_proba(infer_X)[:, 1],
}
# Nearest-seed similarity (max cosine to any training seed), computed in chunks
seed_unit = normalize(seed_X[train_seed])
nn = np.concatenate([(infer_unit[i:i + 5000] @ seed_unit.T).max(axis=1)
                     for i in range(0, len(infer_unit), 5000)])
scores["Nearest-seed cosine (kNN)"] = nn

# -----------------------------------------------------------------------------
# 3. Evaluation across audience sizes
# -----------------------------------------------------------------------------
cuts = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
cos_to_centroid = scores["Cosine to seed centroid"]
rows = []
for model, s in scores.items():
    order = np.argsort(-s, kind="stable")
    for c in cuts:
        top = order[: int(round(c * len(s)))]
        recall = is_holdout[top].sum() / is_holdout.sum()
        rows.append({
            "model": model, "audience_pct": int(c * 100), "audience_size": len(top),
            "holdout_recall": round(recall, 3), "lift": round(recall / c, 2),
            "mean_cosine_to_seed": round(float(cos_to_centroid[top].mean()), 3),
            "true_similar_share": round(float(truly_similar[top].mean()), 3),
        })
res = pd.DataFrame(rows)
res.to_csv(ROOT / "results/audience_quality_by_size.csv", index=False)

baseline = round(float(pub_truly_similar.mean()), 3)
summary = (res[res.audience_pct == 5][["model", "holdout_recall", "lift", "true_similar_share"]]
           .sort_values("holdout_recall", ascending=False))
summary.to_csv(ROOT / "results/model_ranking_at_5pct.csv", index=False)
print("Share of truly similar users in the whole publisher pool:", baseline)
print("\nModel ranking at a 5% audience:\n", summary.to_string(index=False))
print("\nFull table:\n", res.to_string(index=False))

# Rank agreement between the label-free metric and the ground truth
agree = res.groupby("audience_pct").apply(
    lambda g: g["holdout_recall"].rank().corr(g["true_similar_share"].rank(), method="spearman"),
    include_groups=False)
agree.round(2).rename("spearman_rank_agreement").to_csv(ROOT / "results/holdout_vs_truth_agreement.csv")

# -----------------------------------------------------------------------------
# 4. Figures
# -----------------------------------------------------------------------------
colors = dict(zip(scores, ["#4a5568", "#2b6cb0", "#dd6b20", "#319795"]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for m, g in res.groupby("model"):
    ax[0].plot(g.audience_pct, g.holdout_recall, marker="o", label=m, color=colors[m])
    ax[1].plot(g.audience_pct, g.true_similar_share, marker="o", label=m, color=colors[m])
ax[0].plot([c * 100 for c in cuts], cuts, "--", color="grey", label="Random audience")
ax[1].axhline(baseline, ls="--", color="grey", label="Publisher average")
ax[0].set(title="Hold-out recall: share of hidden customers found",
          xlabel="Audience size (% of publisher users)", ylabel="Recall")
ax[1].set(title="Audience quality: share of truly similar users",
          xlabel="Audience size (% of publisher users)", ylabel="Precision (ground truth)")
for a in ax:
    a.grid(alpha=0.3)
ax[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "figures/recall_and_quality_by_audience_size.png", dpi=110)

best = summary.iloc[0]["model"]
g = res[res.model == best]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(g.audience_pct.astype(str) + "%", g.mean_cosine_to_seed, color="#2b6cb0")
for x, v in zip(range(len(g)), g.mean_cosine_to_seed):
    ax.text(x, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
ax.axhline(float(cos_to_centroid[:N_PUB].mean()), ls="--", color="grey", label="Average publisher user")
ax.set(title=f"Similarity to seed customers shrinks as the audience grows\n({best})",
       xlabel="Audience size", ylabel="Mean cosine similarity to seed centroid")
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "figures/similarity_decay.png", dpi=110)
