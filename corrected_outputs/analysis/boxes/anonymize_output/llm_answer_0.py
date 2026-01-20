def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, and odds-ratios for:
      - Age_c (linear age effect)
      - Age_c2 (quadratic age effect)
      - Age_c x Site interaction terms (one per non-reference site dummy)
    Also performs a likelihood-ratio test comparing the fitted (full) model to a reduced
    model without the Age_c x Site interaction terms to test whether developmental
    trajectories differ across sites jointly.

    Returns a dictionary with keys:
      - "object": a nested dict of extracted numeric results
      - "description": a short interpretation of what the numbers mean for the research question
    """
    import numpy as np
    import statsmodels.formula.api as smf

    res = model_output  # expected: statsmodels BinaryResultsWrapper or similar

    # Precompute confidence intervals once (if available)
    try:
        conf_df = res.conf_int()
    except Exception:
        conf_df = None

    # helper to safely extract term stats if present
    def _term_stats(name):
        if name not in res.params.index:
            return None
        coef = float(res.params.loc[name])
        se = float(res.bse.loc[name]) if (hasattr(res, "bse") and name in res.bse.index) else None
        p = float(res.pvalues.loc[name]) if (hasattr(res, "pvalues") and name in res.pvalues.index) else None
        if conf_df is not None and name in conf_df.index:
            ci_vals = conf_df.loc[name].values
            ci = (float(ci_vals[0]), float(ci_vals[1]))
        else:
            ci = (None, None)
        or_ = float(np.exp(coef)) if coef is not None else None
        or_ci = (float(np.exp(ci[0])), float(np.exp(ci[1]))) if (ci[0] is not None and ci[1] is not None) else (None, None)
        return {
            "coef": coef,
            "se": se,
            "p_value": p,
            "ci_95": ci,
            "odds_ratio": or_,
            "odds_ratio_ci_95": or_ci
        }

    # Extract main age effects
    age_stats = _term_stats("Age_c")
    age2_stats = _term_stats("Age_c2")

    # Find interaction terms that represent Age_c x Site.
    # We want names that include 'Age_c' but not 'Age_c2' and are not the main 'Age_c' term.
    inter_names = [n for n in res.params.index if ("Age_c" in n) and (n != "Age_c") and (not n.startswith("Age_c2"))]
    # Filter likely interaction names to those that also mention Site (defensive)
    inter_names = [n for n in inter_names if ("Site" in n) or (":" in n)]
    # Build dictionary of interaction stats
    interactions = {n: _term_stats(n) for n in inter_names}

    # Attempt a likelihood-ratio test comparing full model (res) to reduced model without Age_c x Site interactions.
    lr_test = None
    try:
        # Retrieve the original dataframe used to fit the model
        # statsmodels stores data in res.model.data.frame for many model types
        if hasattr(res.model, "data") and hasattr(res.model.data, "frame") and res.model.data.frame is not None:
            df = res.model.data.frame.copy()
        else:
            # Fall back: try to access exogenous/endogenous arrays to reconstruct minimal dataframe
            raise AttributeError("Original dataframe not available on the fitted model (res.model.data.frame).")

        # Fit reduced model without the Age_c x Site interaction:
        # The formula below mirrors the expected fixed effects from the full model except interactions
        reduced_formula = "MajorityChoice ~ Age_c + Age_c2 + C(Site) + Female + MajorityFirst"
        reduced = smf.logit(formula=reduced_formula, data=df).fit(disp=False)

        # Compare full (res) to reduced: returns (lr_stat, pvalue, df_diff)
        lr_stat, lr_pvalue, lr_df = res.compare_lr_test(reduced)
        lr_test = {"lr_stat": float(lr_stat), "p_value": float(lr_pvalue), "df_diff": int(lr_df)}
    except Exception as e:
        # If something goes wrong (e.g., original data not available), report the error message
        lr_test = {"error": str(e)}

    # Assemble output object
    try:
        conf_int_95 = {idx: (float(ci[0]), float(ci[1])) for idx, ci in (conf_df.iterrows() if conf_df is not None else [])}
    except Exception:
        conf_int_95 = {}

    output_object = {
        "age_term": age_stats,
        "age_squared_term": age2_stats,
        "age_by_site_interactions": interactions,
        "interaction_lr_test": lr_test,
        # Also include full parameter table for reference
        "full_params": {
            "params": res.params.to_dict(),
            "pvalues": res.pvalues.to_dict() if hasattr(res, "pvalues") else {},
            "conf_int_95": conf_int_95
        }
    }

    # Short interpretation description
    desc_lines = [
        "Extracted coefficients, p-values, 95% confidence intervals, and odds ratios for:",
        "- 'Age_c' (linear developmental effect)",
        "- 'Age_c2' (non-linear/quadratic effect)",
        "- Each Age_c x Site interaction term (how the age slope differs by site).",
        "",
        "Interpretation guide:",
        "- If Age_c has a positive, statistically significant coefficient -> overall tendency to rely on the majority increases with age (reported as coefficient and odds ratio).",
        "- If Age_c2 is significant -> the age effect is non-linear (quadratic).",
        "- If one or more Age_c x Site interaction coefficients are significant -> the age-related change differs in that site compared to the reference site.",
        "- The likelihood-ratio test (interaction_lr_test) assesses whether the set of Age_c x Site interactions is jointly different from zero (i.e., whether developmental trajectories differ across sites overall)."
    ]
    description = "\n".join(desc_lines)

    return {"object": output_object, "description": description}