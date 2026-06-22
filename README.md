# 🗺️ SlayTheSpire-StrategyAI

[![License: GPL v3](https://shields.io)](https://gnu.org)
[![Python 3.10+](https://shields.io)](https://python.org)
[![LightGBM](https://shields.io)](https://github.com/lightgbm-org/LightGBM)

An enterprise-grade, data-driven strategy decision engine and automation framework for *Slay the Spire*. By training specialized Gradient Boosted Decision Tree (GBDT) ensembles on tens of thousands of high-Ascension (A20) runs, this system acts as a real-time tactical co-pilot, calculating Softmax probabilities to recommend optimal card rewards or explicit skips.

---

## 🚀 Key Features & Architectural Wins

* **Single-Pass Out-of-Core Data Engine:** Eliminates disk I/O bottlenecks by processing compressed game log archives (`.7z`) via asynchronous stdout subprocess streaming pipelines and high-throughput byte parsing (`orjson`).
* **Run-Insulated Leakage Prevention:** Groups tabular game states using custom chronologically aligned run signatures (floor tracking, class markers). This guarantees that sequential decision snapshots from a single run never bleed across the train/validation split.
* **Act-Specialized Multi-Class Ensemble Routing:** Automatically routes live queries to specialized downstream tree models optimized exclusively for the distinct tactical priorities of **Act 1** (damage/burst scaling), **Act 2** (mitigation/AoE), and **Act 3** (synergy realization).
* **Index-Insulated Validation Harness:** Replaces generic random splits with an evaluation suite tailored to list-wise gaming vectors, mapping candidates to fixed 4-slot screen structures to compute true human-aligned Top-1 accuracy.
* **Resilient Unknown Token Routing:** Mitigates schema drift from prospective game patches by implementing fallback token handling (`unknowncard`, `unknownrelic`) to preserve input dimension footprints during real-time inference.

---

## 📁 Repository Structure

The project implements a modern, modular Python package layout to decouple data ingestion, core optimization, and user-facing interactive interfaces:

```text
SlayTheSpire-StrategyAI/
│
├── .gitignore
├── LICENSE                   # Legally protected under GPLv3 copyleft
├── README.md                 # System documentation
├── setup.py                  # Enables editable local installation via pip
│
├── apps/                     # Interactive presentation layers
│   ├── live_spire_recommender.py  # Inference routing engine and feature builder
│   └── spire_cli.py          # Read-Eval-Print-Loop (REPL) interactive shell
│
├── src/                      # Source library code core
│   └── spire_strategy/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── pipeline.py   # Out-of-core single-pass stream processing
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py    # Multi-class GBDT tree training & validation pipelines
│       └── utils/
│           ├── __init__.py
│           └── profiling.py  # Model complexity and footprint evaluation suite
│
└── tests/                    # Integration verification
    └── test_inference.py     # Mock execution sandbox
```

---

## 🛠️ Installation & Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/VBot2410/SlayTheSpire-StrategyAI.git
   cd SlayTheSpire-StrategyAI
   ```

2. **Initialize and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install Core System Dependencies & Module Package:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
   *Note: Installing with the `-e` flag allows `setup.py` to seamlessly resolve local package absolute imports (`spire_strategy`) across your environment.*

---

## 💻 Technical Usage Guide

### 1. Execute the Streaming Ingestion Pipeline
To process your raw game archive into a vectorized tabular dataset without exhausting system RAM:
```bash
python src/spire_strategy/data/pipeline.py
```

### 2. Train the Multi-Class Tree Ensembles
Fits the specialized LightGBM models across distinct Act intervals, recording true Top-1 choice selection accuracies on unseen runs:
```bash
python src/spire_strategy/training/trainer.py
```

### 3. Launch the Interactive Co-Pilot Console
Launches the persistent-state interactive game tracker. It supports full run initialization, dynamic shop merchant purchases, post-combat reward auditing, and real-time choice evaluations:
```bash
python apps/spire_cli.py
```

---

## 🗺️ Long-Term Development Roadmap

To fulfill the mission of establishing a fully autonomous gameplay agent, development scales along the following tracks:

- [x] Out-of-core data extraction pipeline
- [x] Act-specialized multi-class model ensemble architectures
- [x] REPL interactive console engine state manager
- [ ] **Phase 4:** Expand modeling scope to parse map layout arrays for optimal node-routing path generation.
- [ ] **Phase 5:** Build combat-state tracking arrays to automate card-play selection ordering inside combat encounters.
- [ ] **Phase 6:** Integrate direct memory hook read/write handlers to completely bypass manual text entry interfaces.
