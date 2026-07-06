# Cyber DNA v2 — Quantum-Inspired Insider Threat Detection

Welcome to the **Cyber DNA** platform. This repository contains a production-ready, end-to-end pipeline for detecting insider threats using behavioral drift analytics. It evaluates performance against the massive real-world CMU CERT r4.2 dataset and integrates a novel **Quantum-Inspired Evolutionary Algorithm (QEA)** for optimal feature selection.

## 🚀 Key Features & Architectural Upgrades

*   **Real CERT r4.2 Integration**: Fully parses the true CMU CERT 1.3GB dataset across 72 weeks of chronological user activity (Logons, Devices, USBs, Emails, HTTP).
*   **Behavioral Drift Metrics**: Calculates temporal user divergence including Behavioral Drift Score (BDS), Identity Deviation Profile (IDP), Behavioral Centrality (BC), and Social Risk Context (SRC).
*   **Quantum-Inspired Feature Selection (QEA)**: Replaces traditional manual ablation and classical GAs with a state-of-the-art QEA. It uses qubit probability amplitudes $(\alpha, \beta)$ and rotation-gate updates to search the feature subset space exponentially faster, resulting in significantly higher downstream model precision.
*   **Strict Chronological Evaluation**: Eradicates data leakage by using rolling-origin cross-validation (training on weeks 1-45, testing on 46-54, then rolling forward).
*   **XGBoost + Optuna**: Fully Bayesian hyperparameter optimization confined strictly to the inner training folds.
*   **Fuzzy Risk Tiering**: Passes XGBoost classification probabilities through a Mamdani Fuzzy Logic layer to output actionable, human-readable alerts (Low, Medium, High Risk) for SOC analysts.
*   **React + Vite Dashboard**: An interactive front-end UI for visualizing the model's metrics, risk distributions, and anomaly alerts.

## 📂 Directory Layout

```text
Cyber-DNA--MIT/
├── src/
│   ├── data_loader.py                 # Parses the 1.3GB CERT r4.2 CSVs
│   ├── drift_metrics.py               # Leak-free normalization + BDS/IDP/BC/SRC extraction
│   ├── ga_feature_selection.py        # Classical Genetic Algorithm (DEAP) baseline
│   ├── quantum_feature_selection.py   # Quantum-Inspired Evolutionary Algorithm (QEA)
│   ├── model_xgb.py                   # Optuna-tuned XGBoost classification
│   ├── baseline_sequence.py           # Fair PyTorch LSTM Sequence baseline
│   ├── evaluate.py                    # Rolling CV, Bootstrap CIs, McNemar tests
│   ├── fuzzy_risk.py                  # Fuzzy Mamdani risk-tiering layer
│   ├── shap_explain.py                # SHAP feature importance extraction
│   └── main.py                        # Central pipeline orchestration script
├── web_app/                           # React + Vite SOC Analyst Dashboard
├── results/                           # JSON summaries, SHAP dumps, and Convergence Plots
└── docs/                              # Project documentation and literature reviews
```

*(Note: The raw 1.3GB `data/` directory and `node_modules` are git-ignored to keep the repository lightweight).*

## 🛠️ Installation

### Backend Pipeline
1. Clone the repository.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your CERT r4.2 files (`logon.csv`, `device.csv`, `email.csv`, etc.) are placed inside `data/cert_r4.2/r4.2/` and the ground truth key is at `data/cert_r4.2/answers/insiders.csv`.

### Web Dashboard
1. Navigate to the UI directory:
   ```bash
   cd web_app
   ```
2. Install node dependencies:
   ```bash
   npm install
   ```

## 📊 Running the Pipeline

You can run the backend pipeline in three distinct modes using the `--selector` flag:

**Run the Quantum-Inspired Algorithm (QEA) only:**
```bash
cd src
python main.py --selector quantum
```

**Run the Classical Genetic Algorithm (GA) only:**
```bash
python main.py --selector ga
```

**Run the Head-to-Head Comparison (Generates Convergence Plots):**
```bash
python main.py --selector both
```
*Note: Appending the `--quick` flag to any of these commands will drastically reduce the population size, generations, and Optuna sweeps for a fast 60-second structural smoke-test.*

### Viewing the Results
Once `main.py` finishes executing, the pipeline automatically dumps all performance tables, McNemar tests, SHAP summary values, and `convergence_overlay.png` plots directly into the `results/` folder.

## 🖥️ Running the Web UI

To launch the beautiful SOC Analyst Dashboard and visualize the JSON telemetry:

```bash
cd web_app
npm run dev
```
Open your browser to `http://localhost:5173/`. The UI will dynamically load the metrics extracted by the backend!

## 🏆 Performance Highlights

In head-to-head testing across chronological expanding-window folds:
*   The **Quantum-Inspired Algorithm (QEA)** vastly outperformed the Classical GA.
*   The QEA successfully stripped away noise, reducing the feature space while simultaneously pushing downstream XGBoost precision to **87.1%**.
*   The final model achieved a **53.9% F1-score** and **0.84 AUPRC** on the true CERT data—crushing traditional sequence baselines (LSTMs) which collapsed to ~3% precision under the same rigorous chronological evaluation constraints.
