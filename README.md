# Calibration Methods for Multi-Center Risk Models

**Main contributor:** Aziz Ayed  
**Email:** aziz.ayed@mit.edu  
**Article:** TBD  
**Current release:** 11/15/2025  

This repository contains a minimal, cleaned-up implementation of the calibration workflow used in the accompanying paper. It focuses on:

- Out-of-fold (OOF) nested bootstrap calibration comparison across centers  
- Calibrated incidence at a decision threshold (with bootstrap CIs)  
- Eligibility change under an “oracle” threshold (with bootstrap CIs)  
- Publication-style plots for center-level calibration and performance

---

## Repository structure

```text
calibration-paper/
├─ README.md
├─ requirements.txt
├─ src/
│  └─ calib_paper/
│     ├─ __init__.py
│     ├─ data.py         # toy dataset generator
│     ├─ metrics.py      # calibration + bootstrap CI utilities
│     ├─ bootstrap.py    # OOF nested bootstrap + paired sign test
│     └─ plotting.py     # plotting utilities
├─ scripts/
│  └─ run_toy_example.py # script version of the toy example
└─ notebooks/
   └─ toy_example.ipynb  # Jupyter notebook demo
```

---

## Installation

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

For local development, add `src/` to your Python path:

```bash
export PYTHONPATH=src
```

(or configure this in your IDE).

---

## Toy example

You can run the toy example either as a script or as a notebook.

### Script

```bash
export PYTHONPATH=src
python scripts/run_toy_example.py
```

This will:

- Generate a synthetic multi-center dataset  
- Compute calibrated incidence and eligibility change with bootstrap CIs  
- Run the OOF nested bootstrap calibration comparison  
- Save figures in `figs/`:
  - `toy_absolute_error_by_center.png`
  - `toy_incidence_by_center.png`
  - `toy_three_panel_metrics.png`

### Notebook

```bash
export PYTHONPATH=src
jupyter notebook
```

Then open `notebooks/toy_example.ipynb` and run all cells.  
The notebook mirrors the script but includes inline explanations.

---

## Using the code with your own data

Your dataframe should have at least:

- `center`: center ID  
- `probs`: predicted probabilities  
- `golds`: binary outcomes (0/1)  

Example:

```python
from calib_paper.metrics import compute_eligibility_change_with_ci
from calib_paper.bootstrap import out_of_fold_bootstrap_calibration_comparison_top_centers
from calib_paper.plotting import plot_absolute_bias_summary

target_incidence = df["golds"].mean()

summary_df = out_of_fold_bootstrap_calibration_comparison_top_centers(
    df=df,
    cen_col="center",
    score_col="probs",
    label_col="golds",
    target_incidence=target_incidence,
    get_calibrated_value_at_threshold=None,  # or your preferred function
    use_direct_incidence=True,
)

fig, ax = plot_absolute_bias_summary(
    summary_df,
    methods=["at_op", "intercept", "iso"],
    plot_type="abs_error",
    target_incidence=target_incidence,
)
```

You can copy the exact patterns from the toy example and replace the toy data with your real study cohort.

---

## Citation


> *TBD*

