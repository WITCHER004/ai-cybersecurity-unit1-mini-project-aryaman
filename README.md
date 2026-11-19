# AI-Driven Cyber Threat Awareness and Detection Prototype

Lab Assignment 01 · Unit I: Foundations of AI in Cyber Security · AI in Cyber Security Lab (ENSP355)

| | |
|---|---|
| Name | _your name_ |
| Roll No. | _your roll number_ |
| Course / Section | B.Tech CSE (AIML) / _section_ |

A small prototype that learns from login / network-session records and labels each
event as **normal** or **suspicious**, then checks what goes wrong ethically when
thresholds or training labels change.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── login_events.csv          # synthetic dataset (generated, seed 42)
├── src/
│   ├── config.py                 # paths, seed, feature lists
│   ├── common.py                 # loading, split, preprocessing pipeline, metrics
│   ├── generate_dataset.py       # builds the synthetic dataset
│   ├── task1_check_setup.py      # Task 1: environment check
│   ├── task2_eda.py              # Task 2: threat landscape analysis (EDA)
│   ├── task3_ml_models.py        # Task 3: Logistic Regression + Decision Tree
│   ├── task4_deep_learning.py    # Task 4: MLP (64-32) and ML vs DL comparison
│   ├── task5_ethics.py           # Task 5: thresholds, bias, explainability
│   └── run_all.py                # runs everything in order
├── figures/                      # fig1 ... fig8 (PNG)
└── results/                      # CSV tables for every task
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python src/run_all.py           # whole pipeline (~10 s)
python src/task3_ml_models.py   # or any single task
```

Each task script generates the dataset first if `data/login_events.csv` is missing.

## Dataset

The dataset is synthetic (`src/generate_dataset.py`, seed 42), so the scores show how the
methods behave. Don't compare them with published results on real traffic.

- 20,000 events, 11.9% suspicious (2,381)
- Attack types: brute force (684), phishing login (622), credential stuffing (606), data exfiltration (469)
- 9 numeric features: `hour_of_day`, `failed_attempts`, `requests_per_min`, `bytes_out_mb`,
  `ip_reputation`, `geo_distance_km`, `new_device`, `is_vpn`, `session_minutes`
- 2 categorical features: `country_group`, `device_type`
- Realism: 22% of attacks are "low and slow", 4% of normal events look odd,
  2–4% missing values in `requests_per_min`, `ip_reputation`, `session_minutes`, `device_type`

## Method

- Median imputation and scaling for numeric features. Most-frequent imputation and one-hot
  encoding for categorical features. All steps sit in a scikit-learn `Pipeline`, so test data
  can't leak into training.
- Stratified 75/25 split (15,000 train / 5,000 test).
- Models: Logistic Regression, Decision Tree (depth 6), MLPClassifier (64-32, ReLU, Adam,
  early stopping).
- Metrics: accuracy, plus precision, recall and F1 on the suspicious class, and training time.

## Results (seed 42)

| Model | Accuracy | Precision | Recall | F1 | Missed attacks (FN) | Legit flagged (FP) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.931 | 0.774 | 0.598 | 0.675 | 239 | 104 |
| Decision Tree (depth 6) | 0.938 | 0.758 | 0.704 | 0.730 | 176 | 134 |
| Neural Network (MLP 64-32) | 0.945 | 0.845 | 0.662 | 0.743 | – | – |

- The MLP (3,201 parameters) gives the best F1, but only about 0.013 above the Decision Tree,
  and it trains about 15× slower than Logistic Regression.
- **Threshold trade-off (Logistic Regression):** at a threshold of 0.1, 425 legitimate events are
  blocked and 106 attacks are missed. At 0.9, only 13 legitimate events are blocked, but 521 of
  595 attacks get through.
- **Bias simulation:** 35% of legitimate VPN or international training events (1,015) were
  relabelled as suspicious. The model was then evaluated on the correct labels:
  - The false-positive rate for international users rose from 10.8% to 27.5%.
  - For VPN users it rose from 5.9% to 19.2%.
  - For domestic users without a VPN it stayed around 1%.
  - Overall accuracy moved only from 0.931 to 0.919.
- **Explainability:** permutation importance for the MLP ranks `failed_attempts`,
  `bytes_out_mb` and `requests_per_min` highest.

Exact numbers can differ slightly between library versions; the scripts print and save
the current values in `results/`.

## Outputs

| Figure | Content |
|---|---|
| `fig1_class_balance.png` | Normal vs suspicious, attack-type frequency |
| `fig2_feature_distributions.png` | Feature distributions per class |
| `fig3_suspicious_rate_by_indicator.png` | % suspicious by country, device, VPN, new device |
| `fig4_confusion_matrices.png` | Confusion matrices for LR and DT |
| `fig5_ml_vs_dl.png` | Performance, training time, MLP training curve |
| `fig6_threshold_tradeoff.png` | FP / FN vs threshold, precision vs recall |
| `fig7_bias_simulation.png` | False-positive rate per group, correct vs biased labels |
| `fig8_permutation_importance.png` | MLP permutation importance |

## Limitations

- The data is synthetic and simpler than real traffic.
- The models use one train/test split, with no cross-validation and no hyperparameter tuning.
- Next steps: use a real dataset (e.g. CICIDS2017), add class weighting to improve recall, and
  make per-group fairness checks a standard part of evaluation.

## References

1. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR 12, 2011.
2. NIST, Framework for Improving Critical Infrastructure Cybersecurity.
3. NIST, AI Risk Management Framework (AI RMF 1.0), NIST AI 100-1, 2023.
4. R. Sommer and V. Paxson, "Outside the closed world: On using machine learning for network intrusion detection," IEEE S&P, 2010.
5. Digital Personal Data Protection Act, 2023 (India); Regulation (EU) 2016/679 (GDPR).
