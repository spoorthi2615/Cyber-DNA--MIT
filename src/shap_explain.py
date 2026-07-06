"""
SHAP explainability, to replace/supplement the paper's Figure 2 (raw XGBoost
gain importance only). Gain importance is biased toward high-cardinality /
frequently-split features and doesn't explain individual predictions; SHAP
gives both global (summary plot) and local (per-user-week) explanations,
which reviewers in the explainable-AI-adjacent soft-computing space will
expect for a security-relevant classifier.
"""
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_shap_values(clf, X):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X)
    return shap_values


def save_summary_plot(shap_values, X, out_path):
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_local_explanation(shap_values, index, out_path):
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
