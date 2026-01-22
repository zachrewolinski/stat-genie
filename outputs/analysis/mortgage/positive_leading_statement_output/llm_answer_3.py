def extract_final_answer(model_output):
    """
    Extracts statistics on the effect of the 'Female' indicator from a fitted model output.
    Returns a dictionary with:
      - "object": a dict of numeric extracted values (coef, pvalue, odds_ratio, CI, marginal effect if available)
      - "description": a plain-language interpretation of those values in the context of the task.
    The function is robust to model_output containing either:
      - 'summary_table' (pandas DataFrame indexed by variable names), and/or
      - 'model' (statsmodels results object),
      - 'female_margeff' (statsmodels margins object).
    """
    import numpy as np
    import pandas as pd

    out = {
        "coef": None,
        "pvalue": None,
        "odds_ratio": None,
        "odds_ratio_ci_lower": None,
        "odds_ratio_ci_upper": None,
        "marginal_effect": None,             # average marginal effect (if available)
        "marginal_effect_pvalue": None,
        "source": None
    }

    # Helper to safely set values
    def safe_set(k, v):
        if v is None or (isinstance(v, float) and (np.isnan(v))):
            return
        out[k] = float(v)

    summary_table = model_output.get("summary_table", None)
    model = model_output.get("model", None)
    female_margeff = model_output.get("female_margeff", None)

    # Try to extract from summary_table first (preferred)
    try:
        if isinstance(summary_table, pd.DataFrame):
            # Ensure index contains 'Female'
            if "Female" in summary_table.index:
                row = summary_table.loc["Female"]
                safe_set("coef", row.get("coef", None))
                safe_set("pvalue", row.get("pvalue", None))
                safe_set("odds_ratio", row.get("odds_ratio", None))
                safe_set("odds_ratio_ci_lower", row.get("odds_ratio_ci_lower", None))
                safe_set("odds_ratio_ci_upper", row.get("odds_ratio_ci_upper", None))
                out["source"] = "summary_table"
    except Exception:
        # continue to other extraction methods
        pass

    # If not found in summary_table, try to extract from the statsmodels results object
    if out["coef"] is None and model is not None:
        try:
            # model.params and model.pvalues are common attributes
            if hasattr(model, "params") and "Female" in model.params.index:
                safe_set("coef", model.params["Female"])
                # pvalues might be in .pvalues
                if hasattr(model, "pvalues"):
                    safe_set("pvalue", model.pvalues.get("Female", None))
                # confidence interval via model.conf_int()
                try:
                    ci = model.conf_int().loc["Female"]
                    safe_set("odds_ratio_ci_lower", np.exp(ci[0]))
                    safe_set("odds_ratio_ci_upper", np.exp(ci[1]))
                except Exception:
                    pass
                # compute odds ratio
                try:
                    safe_set("odds_ratio", np.exp(model.params["Female"]))
                except Exception:
                    pass
                out["source"] = "model.params"
        except Exception:
            pass

    # Try to extract marginal effect if available
    if female_margeff is not None:
        # Try multiple ways to get Female's marginal effect
        me = None
        me_p = None
        try:
            # If the object has summary_frame(), use it
            if hasattr(female_margeff, "summary_frame"):
                dfm = female_margeff.summary_frame()
                if "Female" in dfm.index:
                    # typical column name for marg effect is 'dy/dx'
                    if "dy/dx" in dfm.columns:
                        me = dfm.loc["Female", "dy/dx"]
                    else:
                        # fallback to first column
                        me = dfm.iloc[dfm.index.get_loc("Female"), 0]
                    # p-value column might be 'P>|z|' or 'pvalue'
                    for colname in ["P>|z|", "pvalue", "p-value", "p"]:
                        if colname in dfm.columns:
                            me_p = dfm.loc["Female", colname]
                            break
            # If summary_frame not available or didn't contain name, try attributes
            if me is None and hasattr(female_margeff, "margeff"):
                # find index of 'Female' in exog names if possible
                exog_names = None
                try:
                    exog_names = female_margeff.model.exog_names
                except Exception:
                    # maybe nested results
                    try:
                        exog_names = female_margeff._results.model.exog_names
                    except Exception:
                        exog_names = None
                if exog_names is not None and "Female" in exog_names:
                    idx = list(exog_names).index("Female")
                    arr = np.asarray(female_margeff.margeff)
                    if arr.ndim == 1 and idx < arr.size:
                        me = arr[idx]
                else:
                    # fallback to first element
                    arr = np.asarray(female_margeff.margeff)
                    if arr.size >= 1:
                        me = arr.flatten()[0]
        except Exception:
            me = None
            me_p = None

        safe_set("marginal_effect", me)
        safe_set("marginal_effect_pvalue", me_p)

    # Compose a human-readable description using whatever we extracted
    desc_parts = []
    if out["coef"] is not None:
        coef = out["coef"]
        p = out["pvalue"]
        orr = out["odds_ratio"]
        cil = out["odds_ratio_ci_lower"]
        ciu = out["odds_ratio_ci_upper"]

        # Interpret sign and odds ratio
        if orr is not None:
            pct = (orr - 1.0) * 100.0
            desc_parts.append(
                f"Estimated coefficient on Female = {coef:.3f}; odds ratio = {orr:.3f} "
                f"(which corresponds to a {pct:.1f}% change in the odds of acceptance vs. males)."
            )
            if cil is not None and ciu is not None:
                desc_parts.append(f"95% CI for odds ratio = [{cil:.3f}, {ciu:.3f}].")
        else:
            desc_parts.append(f"Estimated coefficient on Female = {coef:.3f} (odds ratio not computed).")

        # Significance
        if p is not None:
            sig_text = "statistically significant" if p < 0.05 else "not statistically significant"
            desc_parts.append(f"P-value = {p:.3g} → {sig_text} at the 5% level.")
    else:
        desc_parts.append("Could not extract coefficient/p-value for 'Female' from the provided model output.")

    # Marginal effect interpretation
    if out["marginal_effect"] is not None:
        me = out["marginal_effect"]
        me_p = out.get("marginal_effect_pvalue", None)
        desc = f"Average marginal effect of Female ≈ {me:.3f} (this is the estimated absolute change in acceptance probability)."
        if me_p is not None:
            sig_text = "statistically significant" if me_p < 0.05 else "not statistically significant"
            desc += f" P-value = {me_p:.3g} → {sig_text}."
        desc_parts.append(desc)
    else:
        desc_parts.append("No average marginal effect for 'Female' was available in the output or it could not be parsed.")

    description = " ".join(desc_parts)

    return {"object": out, "description": description}