# Task-Relevance of Topological Features and the Level Sets of Hidden Neurons

**Sergei Usanov** · Course paper, Faculty of Mathematics, HSE University, Moscow, 2026  
Supervisor: Prof. Vasily Gorbounov

**Paper:** [`paper/course_paper.pdf`](paper/course_paper.pdf) (38 pages)

This repository contains:

- the paper;
- the code behind it;
- the four experiments, as executed notebooks;
- every dataset, trained network and score table those experiments produced.

## Abstract

Interpreting a trained neural network includes a question about its individual
units: what does a single hidden neuron respond to? A natural place to look for
an answer is the data the network was trained on. Such a dataset may contain
topological features, and one may ask whether echoes of those features can be
found in the network itself afterwards.

The question is made computable by fixing the task to binary classification. In
that case relevance splits into two notions. In the first, a feature is relevant
to the task itself when the two classes are interleaved so that separating them
becomes a question about an inside and an outside, a property of the labelled
sample settled before any network is trained. In the second, it is relevant to
the way a given network solved that task when one of its hidden neurons, read as
a scalar function of the input, singles it out in its level sets. This narrower
notion is the one measured here, by structural and homological scores and a
descriptor of the level curve's shape.

Three properties of the data act on those level sets at once: the topological,
geometric and density effects. Controlled settings separate them. Neither the
density of the sample nor architectural capacity beyond what the task requires
manufactures a witness for a task-irrelevant feature. What decides is relevance
in the first sense: the same feature, point for point, is singled out by nearly
every network where it is task-relevant and by none where it is not. The notion
therefore has content. What the level curves follow, though, is the pair of
objects, a feature together with the class inside it, and not the feature alone.

## Main results

Every statement is about ensembles of independently initialised networks: 140
configurations of a dataset and an architecture, 50 networks each, 7,000 networks
in all. Every score is read against a negative control measured in the same
configuration.

1. **The instruments are calibrated before they are used.** A neuron constructed
   so that the feature lies in its level set shows what genuine agreement
   produces. Congruent copies of the feature, placed where the task never looks,
   show what its absence produces: a distribution rather than a number, wide
   enough that no verdict can be read without it.
2. **Task-relevance is visible in individual neurons.** Two datasets carry the
   same circle, point for point, and differ only in whether the task requires
   separating its inside from its outside. Where it does, a witness is found in
   49 networks of 50; where it does not, in 0 of 50. The median best-neuron
   structural score is 0.0349 against 0.3923, and the two ensembles do not
   overlap. The effect is not confined to one neuron: 41.9% of all neurons of the
   task-relevant ensemble pass the threshold.
3. **Neither capacity nor sample density produces the effect.** Networks wider or
   deeper than the task requires, or trained far past convergence, do not witness
   a feature the task does not need. Redistributing the points between the objects
   at a fixed total count changes neither verdict: what was task-irrelevant stays
   unwitnessed, and what was task-relevant stays witnessed. Inside the
   task-relevant configuration, the score is moved by two things: the number of
   points each object is given, and the gap between the two objects.
4. **The level sets follow the pair, not the feature.** The feature can be
   separated from the class it encloses: elongate one of them, turn one across
   the other, or move one off centre. Doing so shows that the shape, orientation
   and position of the separating level curve are set mainly by the enclosed
   class, within limits that the feature sets. A witness can therefore be removed
   without touching the feature at all.
5. **Scope.** All experiments intervene in the data and never in the network. So
   every statement is about agreement between measured quantities, and none says
   that a neuron encodes anything.

## The four experiments

| Paper | Question | Notebook | Scores | Trained networks |
|---|---|---|---|---|
| §7, Experiment 1 | Does a neuron see what the task needs? | [`exp1_sanity_check`](notebooks/exp1_sanity_check.ipynb) | `analysis/results/calibration/sanity_*.csv`, `analysis/results/experiments/exp1_sanity_check*/` | `models/runs/simple_{relevant,irrelevant}/`, runs tagged `tanh_w5_d5_mass-train` |
| §8, Experiment 2 | Is it an artefact of capacity? | [`exp2_arch_sweep`](notebooks/exp2_arch_sweep.ipynb) | `analysis/results/experiments/exp2_arch_sweep/`, `…/exp2_extended/` | `models/runs/simple_{relevant,irrelevant}/`: the other architectures, activations and over-training checkpoints (`_ot*`) |
| §9, Experiment 3 | Is it density? | [`exp4_density_sweep`](notebooks/exp4_density_sweep.ipynb) | `analysis/results/experiments/exp4_density_sweep/` | `models/runs/d4_*/`, 40 dataset configurations |
| §10, Experiment 4 | Whose shape does the separating curve carry? | [`exp5_geometry`](notebooks/exp5_geometry.ipynb) | `analysis/results/experiments/exp5_geometry/` | `models/runs/g5_*/`, 56 dataset configurations |

- The paper numbers the experiments 1–4. The code identifiers (`exp1`, `exp2`,
  `exp4`, `exp5`) keep the numbering of the research log, because the stored
  results and specifications are addressed by them.
- The notebooks are stored executed, with all outputs and figures. Their
  commentary is currently in Russian; English versions are in preparation.

## Repository layout

```
paper/course_paper.pdf      the paper
notebooks/                  the four experiments, executed
src/                        the library the notebooks import
  analysis/                 structural and homological scores, shape descriptor, chromatic TDA,
                            negative controls and calibration, per-experiment readers
  data/                     dataset construction and serialisation
  models/                   the MLP, and a hidden neuron read as a scalar field on the input
  training/                 ensemble training to a fixed test-accuracy gate, over-training
  experiments/              experiment specifications and pipeline stages
  management/               persistence of datasets and trained networks
  visualization/            figures
scripts/                    worker processes started by src/training/parallel_train.py
                            and src/analysis/parallel_eval.py
data/dataset_instances/     every dataset exactly as trained on
models/runs/                every network trained for the four experiments (8,701 runs)
analysis/results/           the scores and the tables derived from them
```

The experiment registry (`src/experiments/registry.py`) also holds
specifications of experiments from the research log that are not part of the
paper.

## Stored artifacts

| Path | Contents |
|---|---|
| `data/dataset_instances/<name>/<name>_0/` | One dataset: points, labels, the train/test split, and the named point sets ("features") the scores compare against |
| `models/runs/<dataset>/<dataset>_0/<run>/` | One trained network: `checkpoint.pt` (weights and architecture), training configuration, metrics, and a reference to its dataset |
| `analysis/results/experiments/<id>/neuron_stats_*.csv[.gz]` | The scores, one row per network × neuron × feature (the target and its negative controls) |
| `analysis/results/experiments/<id>/*.eval_config.json` | Every parameter of the evaluation that produced the table next to it |
| `analysis/results/experiments/<id>/spec.json` | The experiment specification |
| `analysis/results/experiments/<id>/aggregates/`, `…/figures/` | Derived tables and figures |
| `analysis/results/calibration/sanity_*.csv` | The resolution and level-count sweeps that fix the instrument (Experiment 1) |

## Running

Tested with Python 3.13 on Linux, CPU only.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The first cell of every notebook does three things:

- finds the repository root;
- makes the root the working directory;
- puts `src/` on `sys.path`.

The library resolves datasets, weights and scores relative to the root. To use
it outside the notebooks, work from the root with `src/` on the path:

```python
import sys; sys.path.insert(0, "src")
from management.experiment_management import load_model_instance, load_dataset_instance

model, checkpoint = load_model_instance(
    "models/runs/simple_relevant/simple_relevant_0/simple_relevant_0_tanh_w5_d5_mass-train_model0")
data = load_dataset_instance("data/dataset_instances/simple_relevant/simple_relevant_0/",
                             reconstruct_data=False)
```

Most cells read the stored tables. A few cells train networks or score whole
ensembles: the calls to `ensemble_scores` in Experiment 1, and to
`frontier_scan`, `train_pool` and `evaluate_parallel` in Experiment 2. These
cells show how the stored results were produced. They resume from per-shard
caches that are not included, since the merged tables hold the same rows, so
re-running them retrains or re-scores from scratch. That takes hours of CPU
time, and no number in the paper needs it.

## Citation

```bibtex
@misc{usanov2026taskrelevance,
  author = {Usanov, Sergei},
  title  = {Task-Relevance of Topological Features and the Level Sets of Hidden Neurons},
  year   = {2026},
  note   = {Course paper, Faculty of Mathematics, HSE University. Supervisor: V. Gorbounov}
}
```
