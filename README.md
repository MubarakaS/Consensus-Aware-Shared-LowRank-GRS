# Consensus-Aware Shared Low-Rank Group Recommendation

This repository contains a Julia implementation of a **consensus-aware shared low-rank model for group recommendation**.

The model jointly learns user, group, and item representations from three relational views:

\[
R^{UI} \approx UV^\top,
\qquad
A^{GU} \approx \sigma(HU^\top),
\qquad
R^{GI} \approx HV^\top.
\]

Unlike post-hoc group aggregation approaches, each group is represented by a directly learned latent vector \(h_g\). A consensus-aware regularizer then aligns the learned group representation with dominant member-preference directions obtained from the SVD of the member latent matrix.

A second ranking stage calibrates the learned group-item score for top-\(K\) recommendation:

\[
s(g,i)
=
a_g \frac{h_g^\top v_i}{\tau}
+
b_i.
\]

---

## Repository Structure

```text
Consensus-Aware-Shared-LowRank-GRS/
├── README.md
├── Project.toml
├── LICENSE
│
├── src/
│   └── SharedLowRankGRS.jl
│
├── notebooks/
│   └── CAMRa2011_Experiment.ipynb
│
├── scripts/
│   └── run_five_seeds.jl
│
├── results/
│   ├── proposed_model.csv
│   └── reproduced_baselines.csv
│
├── baselines/
│   └── README.md
│
└── data/
    └── README.md
```

---

## Method

### Stage 1: Shared low-rank relational learning

Let

- \(U \in \mathbb{R}^{m \times r}\): user factors,
- \(H \in \mathbb{R}^{p \times r}\): group factors,
- \(V \in \mathbb{R}^{n \times r}\): item factors.

The base objective is

\[
L_{\text{shared}}
=
L_{UI}
+
\alpha L_{GU}
+
\beta L_{GI}
+
\lambda
\left(
\|U\|_F^2
+
\|H\|_F^2
+
\|V\|_F^2
\right).
\]

The group-rating prediction is

\[
\hat r_{g,i}
=
h_g^\top v_i.
\]

### Consensus-aware regularization

For group \(g\), collect the latent member vectors into

\[
U_g
=
P_g \Sigma_g Q_g^\top.
\]

The smallest number of singular directions explaining at least \(\eta\) of the member energy is retained.

For retained direction \(q_{g,k}\), member agreement is

\[
c_{g,k}
=
\frac{
\left|
\sum_{u \in M_g} u_u^\top q_{g,k}
\right|
}{
\sum_{u \in M_g}
\left|
u_u^\top q_{g,k}
\right|
+
\epsilon
}.
\]

The implemented consensus penalty is

\[
L_{\mathrm{cons},g}^{*}
=
\left\|
(I-Q_g^\star Q_g^{\star\top})h_g
\right\|_2^2
+
\mu
\sum_{k=1}^{q_g}
(1-c_{g,k})
(q_{g,k}^\top h_g)^2.
\]

The final Stage-1 objective is

\[
L
=
L_{\mathrm{shared}}
+
\gamma
\sum_g
L_{\mathrm{cons},g}^{*}.
\]

The retained consensus settings are

```text
gamma = 0.01
eta   = 0.90
mu    = 1.0
```

The non-consensus ablation is recovered by setting

```text
gamma = 0.0
```

### Stage 2: Ranking calibration

After Stage 1, the learned group and item factors are fixed and the ranking score is calibrated using

\[
s(g,i)
=
a_g
\frac{h_g^\top v_i}{\tau}
+
b_i,
\]

where

- \(a_g\) is a group-specific scale,
- \(b_i\) is an item bias,
- \(\tau\) is a temperature parameter.

The retained ranking settings include

```text
tau = 0.2
lambda_a = 0.05
candidate_pool = 100
num_negatives = 20
```

Model selection in Stage 1 uses validation RMSE.  
Model selection in Stage 2 uses validation NDCG@10.

---

## Dataset

Experiments are conducted on **CAMRa2011**.

The experimental setup contains:

```text
290 groups
602 users
7710 items
```

The implementation uses user-item interactions, group-item interactions, group membership data, held-out group ratings, and the official sampled negative candidates for ranking evaluation.

Dataset files are not duplicated in this repository unless redistribution terms permit it. See `data/README.md` for the expected directory layout.

---

## Evaluation Metrics

Rating prediction:

- RMSE
- MAE

Ranking:

- HR@5
- HR@10
- HR@20
- NDCG@5
- NDCG@10
- NDCG@20

The official sampled ranking evaluation uses one positive item and 100 negative items per query.

---

## Main Five-Seed Results

Consensus-aware model, seeds 42--46:

| Metric | Mean ± SD |
|---|---:|
| RMSE | **0.267100 ± 0.008173** |
| MAE | **0.191724 ± 0.004050** |
| HR@5 | **0.586897 ± 0.002438** |
| HR@10 | **0.766345 ± 0.004763** |
| HR@20 | **0.877103 ± 0.004400** |
| NDCG@5 | **0.401818 ± 0.001629** |
| NDCG@10 | **0.460082 ± 0.002435** |
| NDCG@20 | **0.488260 ± 0.001345** |

---

## Consensus Ablation

| Metric | Without Consensus | With Consensus |
|---|---:|---:|
| RMSE ↓ | 0.267909 | **0.267100** |
| MAE ↓ | 0.192217 | **0.191724** |
| HR@5 ↑ | **0.587172** | 0.586897 |
| HR@10 ↑ | 0.765931 | **0.766345** |
| HR@20 ↑ | **0.877379** | 0.877103 |
| NDCG@5 ↑ | 0.401392 | **0.401818** |
| NDCG@10 ↑ | 0.459483 | **0.460082** |
| NDCG@20 ↑ | 0.487857 | **0.488260** |

Consensus produces modest improvements in rating accuracy and most ranking measures, while HR@5 and HR@20 remain essentially comparable.

---

## Reproduced Baseline Comparison

The following values correspond to reproduced experiments used for comparison with the proposed method.

| Method | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.5917 | 0.7766 | 0.4037 | 0.4635 |
| AGREE | 0.5876 | 0.7883 | 0.4077 | 0.4727 |
| GroupIM | 0.6110 | 0.7972 | 0.4189 | 0.4796 |
| CubeRec | 0.6331 | 0.8131 | 0.4310 | 0.4899 |
| **Consensus-Aware Shared Low-Rank** | 0.5869 | 0.7663 | 0.4018 | 0.4601 |

These results should be interpreted as a reproducibility and model-complexity comparison rather than a state-of-the-art claim.

---

## Baseline Implementations and References

The following public implementations are useful for reproducing or checking the comparison models.

### AGREE

Attentive Group Recommendation, SIGIR 2018.

Repository:

https://github.com/LianHaiMiao/Attentive-Group-Recommendation

The repository provides an implementation of AGREE and a processed CAMRa2011 setup.

### GroupIM

GroupIM: A Mutual Information Maximization Framework for Neural Group Recommendation, SIGIR 2020.

Repository:

https://github.com/CrowdDynamicsLab/GroupIM

### Group Recommendation Baseline Collection

A useful common implementation repository containing/refactoring several representative methods:

https://github.com/FDUDSDE/WWW2023GroupRecBaselines

It includes:

- AGREE
- GroupIM
- HyperGroup
- HCR
- HHGR
- CubeRec

This is particularly useful when comparing multiple methods under a common codebase.

### ConsRec

ConsRec: Learning Consensus Behind Interactions for Group Recommendation, WWW 2023.

Repository:

https://github.com/FDUDSDE/WWW2023ConsRec

### AlignGroup

AlignGroup: Learning and Aligning Group Consensus with Member Preferences for Group Recommendation, CIKM 2024.

Repository:

https://github.com/Jinfeng-Xu/AlignGroup

AlignGroup reports experiments on CAMRa2011 and Mafengwo.

---

## Important Note on Published vs Reproduced Results

Published numbers and reproduced numbers are kept separate.

Differences in

- interaction filtering,
- train/validation/test preprocessing,
- negative sampling,
- candidate construction,
- random seeds,
- hyperparameter selection,
- implementation version,

can materially affect CAMRa2011 results.

For this reason, this repository does **not** treat published scores from different codebases as strictly protocol-matched comparisons.

---

## Running the Model

Activate the Julia project:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Load the implementation:

```julia
include("src/SharedLowRankGRS.jl")
using .SharedLowRankGRS
```

Run the consensus-aware five-seed experiment:

```julia
results =
    run_five_seeds(
        "data/CAMRa2011";
        seeds=42:46,
        gamma=0.01,
        eta=0.90,
        mu=1.0,
    )
```

Run the no-consensus ablation:

```julia
results_no_consensus =
    run_five_seeds(
        "data/CAMRa2011";
        seeds=42:46,
        gamma=0.0,
    )
```

---

## Notebook

The notebook in

```text
notebooks/CAMRa2011_Experiment.ipynb
```

is intended as a readable experiment walkthrough.

The reusable implementation should remain in

```text
src/SharedLowRankGRS.jl
```

so that the notebook and scripts call the same code rather than maintaining duplicate implementations.

---

## Reproducibility

Recommended experimental settings:

```text
latent rank            = 32
Stage-1 max epochs     = 300
Stage-2 max epochs     = 300
Stage-1 patience       = 15
Stage-2 patience       = 20
Stage-1 learning rate  = 0.005
Stage-2 learning rate  = 0.001
consensus gamma        = 0.01
consensus eta          = 0.90
consensus mu           = 1.0
ranking tau            = 0.2
seeds                  = 42, 43, 44, 45, 46
```

Exact experiment settings should be kept in a single configuration object in the source code.

---

## Results Files

Recommended result files:

```text
results/proposed_model.csv
results/reproduced_baselines.csv
```

`proposed_model.csv` should contain one row per random seed.

`reproduced_baselines.csv` should document:

- method,
- repository used,
- preprocessing source,
- seed,
- HR@K,
- NDCG@K,
- runtime if available,
- compatibility changes if any.

---

## Citation

A formal citation will be added after publication.

For now, if you use this repository, please cite the corresponding paper/preprint once available.

```bibtex
@article{sharedlowrankgrs,
  title   = {Shared Low-Rank Relational Learning for Group Recommendation with Consensus-Aware Regularisation},
  author  = {To be added},
  year    = {2026}
}
```

---

## License

Choose a license before public release.

For research code, common options include:

- MIT License
- BSD 3-Clause License
- Apache License 2.0

Dataset licensing remains separate from the license of this repository.

---

## Acknowledgements

The baseline comparison makes use of public implementations and released research code from the group recommender systems community. Please cite the corresponding original papers when using those methods.
