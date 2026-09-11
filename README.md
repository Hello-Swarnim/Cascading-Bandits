# Cascading Bandits

Reproduction and extension of **["Cascading Bandits: Learning to Rank in the Cascade Model"](main_paper.pdf)** (Kveton, Szepesvári, Wen & Ashkan, ICML 2015) — a project for **CS-728: Theory of Multi-Armed Bandits**, IIT Kanpur.

The paper studies online learning to rank under the **cascade click model**: a user scans a ranked list of `K` items from a pool of `L`, examining each in order and stopping at the first one they click. This repo reproduces the paper's core algorithms and regret guarantees, then stress-tests them under a more realistic, misspecified click model.

## What's here

- **[`project.pdf`](project.pdf)** — the accompanying report: paper summary, theoretical contributions (including extended proofs of the gap-dependent upper regret bound and the lower bound), and experimental analysis.
- **[`main_paper.pdf`](main_paper.pdf)** — the original paper.
- **[`Codebase.ipynb`](Codebase.ipynb)** — all implementation and experiments.

## Implemented algorithms

- **CascadeUCB1** — UCB1-style confidence radii (`√(1.5·log t / T(e))`) on per-item attraction estimates.
- **CascadeKL-UCB** — tighter, KL-divergence-based confidence bounds (via `scipy.optimize.brentq` / Newton iteration) on Bernoulli attraction probabilities.
- **RankedKL-UCB** — position-aware variant that maintains separate KL-UCB estimates per `(position, item)` pair rather than a single per-item estimate.

Each algorithm is evaluated in two settings:

1. **Cascade model (as specified in the paper)** — `simulate_cascade`, where the user clicks the first attractive item.
2. **Misspecified Dynamic Bayesian Network (DBN) click model** — `dbn_step`, which adds a satisfaction probability `ν` and continuation probability `γ`, so a click doesn't necessarily stop the session. This tests robustness when the algorithms' cascade assumptions don't actually hold.

## Experimental setup

- Benchmark problem: `BLB(L=32, K=8, p=0.2, Δ=0.15)`, following the paper's convention.
- Averaged over **20 independent runs of 100,000 steps** per algorithm, across **9+ configurations**.
- Metric: cumulative regret against the optimal `K`-item list, with mean ± standard error plotted over time.

### Key findings
- All three algorithms reproduce the paper's **logarithmic regret scaling**.
- Under the misspecified DBN click model, **CascadeKL-UCB** still learns the optimal ranking despite the broken cascade assumption, achieving **~3× lower regret** than RankedKL-UCB.

## Running it

The notebook is self-contained — no external data required.

```bash
pip install numpy scipy matplotlib
jupyter notebook Codebase.ipynb
```

Running a cell's `__main__` block (or calling `run_experiment(...)` directly) simulates the chosen algorithms and saves a cumulative-regret plot (e.g. `cascading_bandits_plot_32_8_0.2_0.15_latest.png`).

## Reference

Kveton, B., Szepesvári, C., Wen, Z., & Ashkan, A. (2015). *Cascading Bandits: Learning to Rank in the Cascade Model.* ICML 2015.
