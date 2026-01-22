def extract_final_answer(model_output):
    """
    Extracts the effect of 'is_human' from a fitted model_output produced by the
    provided modeling function and returns a structured answer.

    Returns a dictionary with keys:
      - "object": a dict with numeric results (coef, odds_ratio, 95% CI, p_value, significance)
      - "description": a short plain-language interpretation answering whether modern humans
                       have higher AMTL frequency after adjusting for covariates.
    """
    import numpy as np

    try:
        mo = model_output

        # Try common structures produced by the modeling function:
        results = None
        or_row = None

        if isinstance(mo, dict):
            results = mo.get('results', None)
            # Prefer the precomputed odds_ratio_table / is_human_summary if available
            or_table = mo.get('odds_ratio_table', None)
            if or_table is not None and 'is_human' in or_table.index:
                or_row = or_table.loc['is_human']
                coef = float(or_row['coef'])
                odds_ratio = float(or_row['odds_ratio'])
                ci_lower = float(or_row['ci_lower'])
                ci_upper = float(or_row['ci_upper'])
            elif 'is_human_summary' in mo:
                row = mo['is_human_summary']
                coef = float(row['coef'])
                odds_ratio = float(row['odds_ratio'])
                ci_lower = float(row['ci_lower'])
                ci_upper = float(row['ci_upper'])
            else:
                # Fall back to extracting from results object if present
                if results is None:
                    raise ValueError("Cannot find 'odds_ratio_table' or 'results' in model_output.")
                coef = float(results.params['is_human'])
                odds_ratio = float(np.exp(coef))
                ci = results.conf_int().loc['is_human']
                ci_lower = float(np.exp(ci[0]))
                ci_upper = float(np.exp(ci[1]))
        else:
            # If model_output is a statsmodels results object itself
            results = mo
            coef = float(results.params['is_human'])
            odds_ratio = float(np.exp(coef))
            ci = results.conf_int().loc['is_human']
            ci_lower = float(np.exp(ci[0]))
            ci_upper = float(np.exp(ci[1]))

        # Extract p-value if possible
        p_value = None
        if results is not None:
            try:
                p_value = float(results.pvalues['is_human'])
            except Exception:
                p_value = None

        # Determine statistical significance: prefer p-value, else use CI excluding 1
        if p_value is not None:
            significant = (p_value < 0.05)
        else:
            significant = not (ci_lower <= 1.0 <= ci_upper)

        # Draw final conclusion (relative increase/decrease)
        if significant and odds_ratio > 1.0:
            conclusion = "Yes"
            interpretation_short = "modern humans have significantly higher odds of AMTL."
        elif significant and odds_ratio < 1.0:
            conclusion = "Yes (lower)"
            interpretation_short = "modern humans have significantly lower odds of AMTL."
        elif not significant:
            conclusion = "No / Inconclusive"
            interpretation_short = "there is no statistically significant difference in AMTL odds for modern humans."
        else:
            conclusion = "Inconclusive"
            interpretation_short = "could not determine a clear effect."

        # Prepare returned numeric object
        numeric_object = {
            'coef_logit': coef,
            'odds_ratio': odds_ratio,
            '95ci_odds_ratio': (ci_lower, ci_upper),
            'p_value': p_value,
            'significant': bool(significant),
            'conclusion': conclusion
        }

        # Compose human-readable description
        pval_str = f"{p_value:.3g}" if p_value is not None else "NA"
        description = (
            f"Effect of is_human on AMTL (adjusted for age, prob_male, and tooth class):\n"
            f"  - Log-odds coefficient = {coef:.4f}\n"
            f"  - Odds ratio = {odds_ratio:.3f} (95% CI: {ci_lower:.3f} to {ci_upper:.3f})\n"
            f"  - p-value = {pval_str}\n"
            f"Interpretation: {interpretation_short} (conclusion: {conclusion})."
        )

        return {'object': numeric_object, 'description': description}

    except Exception as e:
        return {
            'object': None,
            'description': f"Failed to extract is_human effect from model_output: {e}"
        }