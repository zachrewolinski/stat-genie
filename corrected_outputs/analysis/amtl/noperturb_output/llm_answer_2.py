def extract_final_answer(model_output):
    """
    Extracts genus-comparison statistics from the cluster-robust GLM result object.

    Returns a dictionary with:
      - "object": a dict keyed by non-reference genus (e.g., 'Pan', 'Pongo', 'Papio')
                  containing coefficient (log-odds), cluster-robust SE, p-value,
                  95% CI on log-odds, odds ratio and its 95% CI, and a short
                  verdict about whether Homo sapiens has higher AMTL than that genus.
      - "description": brief explanation of the returned fields and how to
                       interpret them (sign/direction and significance).
    """
    import numpy as np
    import pandas as pd
    import re

    # Pull core pieces from the model output
    try:
        params = model_output.params  # pandas Series
        pvals = model_output.pvalues  # pandas Series
        bse_arr = np.asarray(model_output.bse)  # numpy array or array-like
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not read params/bse/pvalues from model_output: {e}"
        }

    # Align bse with params index
    try:
        bse = pd.Series(bse_arr, index=params.index)
    except Exception:
        # Fallback: if lengths mismatch, try to truncate or pad (best-effort)
        if len(bse_arr) == len(params):
            bse = pd.Series(bse_arr, index=params.index)
        else:
            return {
                "object": None,
                "description": "Mismatch between params and bse lengths; cannot align standard errors."
            }

    # Identify genus contrast terms (non-reference genera)
    # Typical param name: 'C(genus, Treatment(reference="Homo sapiens"))[T.Pan]'
    genus_term_names = [name for name in params.index if 'genus' in name]
    if not genus_term_names:
        # Try alternative pattern: any param containing '[T.' (factor levels)
        genus_term_names = [name for name in params.index if '[T.' in name and ('Pan' in name or 'Pongo' in name or 'Papio' in name)]

    if not genus_term_names:
        return {
            "object": None,
            "description": "No genus terms found in model parameters. Ensure the model used 'C(genus, ...)' and that 'Homo sapiens' was the reference."
        }

    results = {}
    z_crit = 1.96  # approx for 95% CI using normal approx (Wald)

    for term in genus_term_names:
        coef = float(params.loc[term])
        se = float(bse.loc[term])
        pval = float(pvals.loc[term]) if term in pvals.index else None
        ci_lower = coef - z_crit * se
        ci_upper = coef + z_crit * se
        or_point = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower))
        or_ci_upper = float(np.exp(ci_upper))

        # Extract genus label from the parameter name
        m = re.search(r'\[T\.([^\]]+)\]', term)
        if m:
            genus_label = m.group(1)
        else:
            # fallback: take trailing part after last dot or bracket
            parts = re.split(r'[\.\[\]]+', term)
            genus_label = parts[-1] if parts else term

        # Interpretation relative to Homo sapiens (reference):
        # coef > 0 -> genus has higher AMTL than Homo sapiens
        # coef < 0 -> genus has lower AMTL than Homo sapiens
        if pval is not None:
            signif = (pval < 0.05)
        else:
            signif = None

        if coef < 0:
            direction = "Homo sapiens has higher AMTL"
        elif coef > 0:
            direction = f"{genus_label} has higher AMTL"
        else:
            direction = "No difference in AMTL"

        significance_text = ("statistically significant (p < 0.05)" if signif else
                             "not statistically significant" if signif is not None else
                             "p-value unavailable")

        results[genus_label] = {
            "term_name": term,
            "coef_log_odds": coef,
            "cluster_robust_se": se,
            "p_value": pval,
            "95%CI_log_odds": (ci_lower, ci_upper),
            "odds_ratio": or_point,
            "95%CI_odds_ratio": (or_ci_lower, or_ci_upper),
            "direction_interpretation": direction,
            "significance": significance_text,
            "conclusion_brief": f"{direction}; {significance_text}."
        }

    description = (
        "For each non-human genus (rows), the function returns the model coefficient "
        "(log-odds) comparing that genus to the reference 'Homo sapiens' (so coefficient = "
        "genus minus Homo). A negative coefficient means the non-human genus has lower AMTL "
        "than Homo sapiens (i.e., Homo sapiens has higher AMTL). p-values and cluster-robust "
        "SEs were used to assess statistical significance. Odds ratios (exp(coef)) and their "
        "95% CIs are provided for easier interpretation on the multiplicative scale."
    )

    return {"object": results, "description": description}