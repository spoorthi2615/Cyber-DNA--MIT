"""
Fuzzy risk-tiering layer (soft-computing interpretability component).

XGBoost outputs a calibrated probability. On its own that's just a number an
analyst has to threshold. Here we build a small Mamdani fuzzy inference
system that takes (model probability, BDS, IDP) as crisp inputs and outputs
a fuzzy "insider risk" tier (Low / Medium / High), which is what a SOC
analyst actually wants to triage on. This gives the paper a second,
independent soft-computing contribution (alongside GA feature selection)
that is squarely in scope for a soft-computing venue, and it directly
extends the "SOC deployment" discussion already in Section 7.5.
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_fuzzy_system():
    prob = ctrl.Antecedent(np.linspace(0, 1, 101), "model_prob")
    bds = ctrl.Antecedent(np.linspace(0, 1, 101), "bds_norm")   # BDS rescaled to [0,1] beforehand
    idp = ctrl.Antecedent(np.linspace(0, 1, 101), "idp")
    risk = ctrl.Consequent(np.linspace(0, 1, 101), "risk")

    prob.automf(3, names=["low", "med", "high"])
    bds.automf(3, names=["low", "med", "high"])
    idp["low"] = fuzz.trimf(idp.universe, [0, 0, 0.5])
    idp["med"] = fuzz.trimf(idp.universe, [0.2, 0.5, 0.8])
    idp["high"] = fuzz.trimf(idp.universe, [0.5, 1, 1])

    risk["low"] = fuzz.trimf(risk.universe, [0, 0, 0.4])
    risk["medium"] = fuzz.trimf(risk.universe, [0.2, 0.5, 0.8])
    risk["high"] = fuzz.trimf(risk.universe, [0.6, 1, 1])

    rules = [
        ctrl.Rule(prob["high"], risk["high"]),
        ctrl.Rule(prob["low"] & bds["low"], risk["low"]),
        ctrl.Rule(prob["med"] & bds["high"], risk["high"]),
        ctrl.Rule(prob["med"] & bds["med"], risk["medium"]),
        ctrl.Rule(prob["low"] & bds["high"] & idp["low"], risk["medium"]),
        ctrl.Rule(prob["low"] & idp["high"], risk["low"]),
        ctrl.Rule(prob["med"] & bds["low"], risk["medium"]),
    ]
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def score_risk_tier(sim, model_prob, bds_norm, idp_val):
    sim.input["model_prob"] = float(np.clip(model_prob, 0, 1))
    sim.input["bds_norm"] = float(np.clip(bds_norm, 0, 1))
    sim.input["idp"] = float(np.clip(idp_val, 0, 1))
    sim.compute()
    score = sim.output["risk"]
    tier = "Low" if score < 0.4 else ("Medium" if score < 0.7 else "High")
    return score, tier


def batch_risk_tiers(model_probs, bds_values, idp_values):
    """Vectorized-ish helper: builds one sim and loops (skfuzzy sims are stateful/cheap)."""
    sim = build_fuzzy_system()
    bds_norm = (bds_values - bds_values.min()) / (bds_values.max() - bds_values.min() + 1e-9)
    scores, tiers = [], []
    for p, b, i in zip(model_probs, bds_norm, idp_values):
        s, t = score_risk_tier(sim, p, b, i)
        scores.append(s)
        tiers.append(t)
    return np.array(scores), np.array(tiers)
