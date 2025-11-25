def extract_final_answer(model_output):
    """
    Extract key statistics from the model output returned by the provided `model` function.

    Returns a dict with:
      - "object": a dict of extracted numeric results for each model found (nb_model, gender_model, damage_model)
      - "description": a short plain-language interpretation about whether the evidence supports
                       the hypothesis that more-feminine names are associated with different outcomes.

    The function handles either:
      - a dict with keys 'nb_model', 'gender_model', 'damage_model' (as returned by the model function), or
      - a single fitted model object (assumed to be the negative-binomial / primary model).
    """
    import numpy as np
    import math

    # Helper to safely extract a parameter-related summary from a fitted results object
    def summarize_param(res, param_name):
        out = {"present": False, "coef": None, "pvalue": None, "ci_lower": None, "ci_upper": None}
        if res is None:
            return out
        # many statsmodels result wrappers expose params, pvalues, conf_int()
        try:
            params = getattr(res, "params", None)
            if params is None:
                return out
            # params might be a pandas Series or numpy array with index
            if param_name not in params.index:
                return out
            coef = float(params[param_name])
            out["present"] = True
            out["coef"] = coef

            # p-value
            pvals = getattr(res, "pvalues", None)
            if pvals is not None and param_name in pvals.index:
                out["pvalue"] = float(pvals[param_name])

            # confidence interval
            try:
                ci = res.conf_int()
                if param_name in ci.index:
                    out["ci_lower"] = float(ci.loc[param_name, 0])
                    out["ci_upper"] = float(ci.loc[param_name, 1])
            except Exception:
                # some result objects may provide conf_int as a method taking alpha
                try:
                    ci = res.conf_int()
                    # If returned as ndarray without index, try to find param position
                    # fallback: skip
                except Exception:
                    pass
        except Exception:
            # defensive: if anything fails, return what we have
            pass
        return out

    # Normalize input
    models = {}
    if model_output is None:
        return {"object": None, "description": "No model output provided."}
    if isinstance(model_output, dict):
        models = model_output
    else:
        # assume it's the primary model
        models = {"nb_model": model_output}

    summary = {}

    # 1) Primary NB model: effect of masfem_center on alldeaths
    nb_res = models.get("nb_model", None)
    nb_sum = summarize_param(nb_res, "masfem_center")
    if nb_sum["present"]:
        # incidence rate ratio (IRR) and CI (exp of coef and CI) if GLM with log link
        try:
            irr = math.exp(nb_sum["coef"])
            irr_ci_lower = math.exp(nb_sum["ci_lower"]) if nb_sum["ci_lower"] is not None else None
            irr_ci_upper = math.exp(nb_sum["ci_upper"]) if nb_sum["ci_upper"] is not None else None
        except Exception:
            irr = irr_ci_lower = irr_ci_upper = None
        nb_sum.update({"irr": irr, "irr_ci_lower": irr_ci_lower, "irr_ci_upper": irr_ci_upper})
    summary["nb_model"] = nb_sum

    # 2) Gender (binary) model: effect of gender_female on alldeaths
    gender_res = models.get("gender_model", None)
    gender_sum = summarize_param(gender_res, "gender_female")
    if gender_sum["present"]:
        try:
            irr = math.exp(gender_sum["coef"])
            irr_ci_lower = math.exp(gender_sum["ci_lower"]) if gender_sum["ci_lower"] is not None else None
            irr_ci_upper = math.exp(gender_sum["ci_upper"]) if gender_sum["ci_upper"] is not None else None
        except Exception:
            irr = irr_ci_lower = irr_ci_upper = None
        gender_sum.update({"irr": irr, "irr_ci_lower": irr_ci_lower, "irr_ci_upper": irr_ci_upper})
    summary["gender_model"] = gender_sum

    # 3) Damage OLS: effect of masfem_center on log_ndam15
    damage_res = models.get("damage_model", None)
    damage_sum = summarize_param(damage_res, "masfem_center")
    if damage_sum["present"]:
        # For log outcome, interpret percent change approx = 100*(exp(beta)-1)
        try:
            pct_change = (math.exp(damage_sum["coef"]) - 1) * 100
            pct_ci_lower = (math.exp(damage_sum["ci_lower"]) - 1) * 100 if damage_sum["ci_lower"] is not None else None
            pct_ci_upper = (math.exp(damage_sum["ci_upper"]) - 1) * 100 if damage_sum["ci_upper"] is not None else None
        except Exception:
            pct_change = pct_ci_lower = pct_ci_upper = None
        damage_sum.update({"pct_change_log_ndam15": pct_change,
                           "pct_ci_lower": pct_ci_lower, "pct_ci_upper": pct_ci_upper})
    summary["damage_model"] = damage_sum

    # Interpret the primary test (nb_model) relative to the hypothesis.
    # Hypothesis: More-feminine names -> fewer precautionary measures -> more fatalities.
    interpretation = ""
    if not nb_sum["present"]:
        interpretation = ("Primary model did not contain a parameter named 'masfem_center', "
                          "so no direct test of continuous femininity on fatalities is available.")
    else:
        coef = nb_sum["coef"]
        p = nb_sum["pvalue"]
        irr = nb_sum.get("irr", None)
        # Decision rule
        if p is not None:
            if p < 0.05:
                # significant
                if coef > 0:
                    interpretation = (f"The coefficient on masfem_center is positive (coef={coef:.4f}, p={p:.3g}), "
                                      f"IRR={irr:.3f} (95% CI ≈ [{nb_sum['irr_ci_lower']:.3f}, {nb_sum['irr_ci_upper']:.3f}]). "
                                      "This is consistent with the hypothesis: more-feminine names are associated with MORE fatalities.")
                else:
                    interpretation = (f"The coefficient on masfem_center is negative (coef={coef:.4f}, p={p:.3g}), "
                                      f"IRR={irr:.3f} (95% CI ≈ [{nb_sum['irr_ci_lower']:.3f}, {nb_sum['irr_ci_upper']:.3f}]). "
                                      "This is statistically significant but in the opposite direction of the hypothesis: "
                                      "more-feminine names are associated with FEWER fatalities.")
            else:
                interpretation = (f"The coefficient on masfem_center is {coef:.4f} (p={p:.3g}), IRR={irr:.3f} "
                                  f"(95% CI ≈ [{nb_sum.get('irr_ci_lower')}, {nb_sum.get('irr_ci_upper')}]). "
                                  "The effect is not statistically significant at conventional levels, so the evidence is inconclusive.")
        else:
            interpretation = (f"The coefficient on masfem_center is {coef:.4f}. p-value unavailable; "
                              "cannot determine statistical significance from the provided model object.")

    return {"object": summary, "description": interpretation}