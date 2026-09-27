# Consensus-Aware Shared Low-Rank Group Recommendation

This repository contains a Julia implementation of a
**consensus-aware shared low-rank model for group recommendation**.

The method jointly learns user, group, and item representations from
user-item, group-user, and group-item relations.

Unlike approaches that construct group representations through post-hoc
aggregation of member preferences, the proposed method directly learns a
latent representation for each group. A consensus-aware regularizer then
encourages the learned group representation to remain consistent with the
dominant preference directions of its members.

The model is trained in two stages:

**Stage 1: Consensus-Aware Shared Low-Rank Representation Learning**

User, group, and item representations are learned jointly from the three
relational views. Group-member consensus is incorporated through an
SVD-based regularization mechanism.

**Stage 2: Low-Rank Ranking Refinement**

The learned group and item representations are retained and the low-rank
group-item scores are refined for top-K recommendation using a
group-specific scale and item bias.

---

## Running the Model

From the repository root, load the Julia implementation:

```julia
include("src/SharedLowRankGRS.jl")
using .SharedLowRankGRS
```

---

## Repository Structure

```text
Consensus-Aware-Shared-LowRank-GRS/
├── README.md
├── src/
│   └── SharedLowRankGRS.jl
├── notebooks/
│   └── Consensus_Aware_Shared_LowRank.ipynb
└── data/
    └── README.md
```