def extract_final_answer(model_output):
    """
    Extracts key statistics (coef, IRR, 95% CI, p-value) for the predictors of interest
    (LiveBait and Camper) from the model output returned by the modeling function.

    Returns a dictionary with:
      - "object": dict with per-variable statistics (or None if no results)
      - "description": human-readable interpretation of what was returned and
                       what it implies in context.

    Handles cases where the model was not fit (results is None and/or irr_table empty).
    """
    import numpy as np
    import pandas as pd

    # Basic validation
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dictionary as returned by the model function.")

    results = model_output.get('results', None)
    irr_table = model_output.get('irr_table', None)

    # Case: no fitted results and empty/no irr_table
    if results is None and (irr_table is None or (isinstance(irr_table, pd.DataFrame) and irr_table.empty)):
        return {
            "object": None,
            "description": (
                "No model estimates available: the model output contains no fitted results "
                "and the IRR table is empty. This typically happens when the input dataframe "
                "was empty or model fitting failed. Please provide a prepared, non-empty dataset "
                "and re-run the model to obtain estimated effects of LiveBait and Camper on catch rate."
            )
        }

    # If an IRR table was provided and non-empty, prefer it (it's ready-to-use)
    if isinstance(irr_table, pd.DataFrame) and not irr_table.empty:
        # Select rows of interest if present
        variables_of_interest = ['LiveBait', 'Camper']
        available = {v: (v in irr_table.index) for v in variables_of_interest}

        extracted = {}
        for v in variables_of_interest:
            if available[v]:
                row = irr_table.loc[v]
                # Ensure values are Python scalars
                extracted[v] = {
                    'coef': float(row['coef']),
                    'IRR': float(row['IRR']),
                    'IRR_ci_lower': float(row['IRR_ci_lower']),
                    'IRR_ci_upper': float(row['IRR_ci_upper']),
                    'pvalue': float(row['pvalue'])
                }
            else:
                extracted[v] = None

        # Build description with brief interpretation
        interpretations = []
        for v in variables_of_interest:
            stats = extracted[v]
            if stats is None:
                interpretations.append(f"{v}: not present in the model output.")
            else:
                irr = stats['IRR']
                p = stats['pvalue']
                direction = "higher" if irr > 1 else ("lower" if irr < 1 else "no change")
                signif = "statistically significant" if (p < 0.05) else "not statistically significant"
                interpretations.append(
                    f"{v}: IRR={irr:.3f} (95% CI {stats['IRR_ci_lower']:.3f}–{stats['IRR_ci_upper']:.3f}), "
                    f"p={p:.3g}. Interpretation: groups with {v} have a {direction} catch rate per hour; "
                    f"this effect is {signif} at alpha=0.05."
                )

        description = (
            "Extracted incidence-rate-ratio (IRR) results for predictors affecting fish caught per hour. "
            "IRR > 1 means a higher catch rate per hour; IRR < 1 means a lower catch rate per hour. "
            "See per-variable interpretations below:\n- " + "\n- ".join(interpretations)
        )

        return {
            "object": extracted,
            "description": description
        }

    # Otherwise, try to extract from the fitted results object (statsmodels-like)
    if results is not None:
        try:
            params = results.params  # pandas Series
            conf = results.conf_int()
            pvalues = results.pvalues

            vars_of_interest = ['LiveBait', 'Camper']
            extracted = {}
            for v in vars_of_interest:
                if v in params.index:
                    coef = float(params.loc[v])
                    irr = float(np.exp(coef))
                    ci_low = float(np.exp(conf.loc[v].iloc[0]))
                    ci_high = float(np.exp(conf.loc[v].iloc[1]))
                    p = float(pvalues.loc[v])
                    extracted[v] = {
                        'coef': coef,
                        'IRR': irr,
                        'IRR_ci_lower': ci_low,
                        'IRR_ci_upper': ci_high,
                        'pvalue': p
                    }
                else:
                    extracted[v] = None

            # Build description similar to above
            interpretations = []
            for v in vars_of_interest:
                stats = extracted[v]
                if stats is None:
                    interpretations.append(f"{v}: not included in fitted model.")
                else:
                    irr = stats['IRR']
                    p = stats['pvalue']
                    direction = "higher" if irr > 1 else ("lower" if irr < 1 else "no change")
                    signif = "statistically significant" if (p < 0.05) else "not statistically significant"
                    interpretations.append(
                        f"{v}: IRR={irr:.3f} (95% CI {stats['IRR_ci_lower']:.3f}–{stats['IRR_ci_upper']:.3f}), "
                        f"p={p:.3g}. Interpretation: groups with {v} have a {direction} catch rate per hour; "
                        f"this effect is {signif} at alpha=0.05."
                    )

            description = (
                "Extracted model coefficients and converted them to incidence-rate-ratios (IRRs) for the predictors. "
                "IRR > 1 indicates higher fish caught per hour; IRR < 1 indicates lower. "
                "Per-variable interpretations:\n- " + "\n- ".join(interpretations)
            )

            return {
                "object": extracted,
                "description": description
            }

        except Exception as e:
            return {
                "object": None,
                "description": f"Model results exist but could not be parsed: {e}"
            }

    # Fallback
    return {
        "object": None,
        "description": "No usable model information found in model_output."
    }