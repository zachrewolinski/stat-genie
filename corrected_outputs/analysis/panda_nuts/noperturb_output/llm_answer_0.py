def extract_final_answer(model_output):
    """
    Extract coefficients, cluster-robust inference, and interpret effects from the provided
    model_output dictionary. Expects model_output to contain a key
    'glm_nb_clustered_result' (preferred) or 'glm_nb_result' with a statsmodels GLMResultsWrapper.

    Returns:
      {
        "object": pd.DataFrame with rows for predictors and columns:
          ['coef', 'bse', 'pvalue', 'ci_lower', 'ci_upper', 'IRR', 'IRR_ci_lower', 'IRR_ci_upper']
        "description": str summary interpreting the effect of age_c, sex_m, and help_y
      }
    """
    import numpy as np
    import pandas as pd

    # Choose clustered result if available, otherwise fall back to plain result
    res = None
    if isinstance(model_output, dict):
        if 'glm_nb_clustered_result' in model_output and model_output['glm_nb_clustered_result'] is not None:
            res = model_output['glm_nb_clustered_result']
        elif 'glm_nb_result' in model_output and model_output['glm_nb_result'] is not None:
            res = model_output['glm_nb_result']
    else:
        # if something else was passed, try to use it directly
        res = model_output

    if res is None:
        raise ValueError("No valid model result found in model_output.")

    # Extract basic statistics
    try:
        params = res.params.copy()
    except Exception as e:
        raise RuntimeError("Could not extract params from model result: " + str(e))

    # Robust objects from get_robustcov_results typically expose bse, pvalues, conf_int
    # If those do not exist on the chosen object, attempt to fall back to the original fit
    try:
        bse = res.bse.copy()
    except Exception:
        bse = pd.Series(np.nan, index=params.index)

    try:
        pvalues = res.pvalues.copy()
    except Exception:
        pvalues = pd.Series(np.nan, index=params.index)

    try:
        ci = res.conf_int()
        # conf_int returns ndarray or DataFrame; ensure DataFrame with same index/order as params
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            # older statsmodels sometimes returns (lower, upper)
            ci = pd.DataFrame({0: ci[0], 1: ci[1]}, index=params.index)
        else:
            ci = pd.DataFrame(ci, index=params.index)
    except Exception:
        # If CI can't be computed from this object, set to NaN
        ci = pd.DataFrame(np.nan, index=params.index, columns=[0, 1])

    # Build results DataFrame for predictors of interest only
    predictors = ['age_c', 'sex_m', 'help_y']
    rows = []
    for pred in predictors:
        if pred in params.index:
            coef = float(params.loc[pred])
            se = float(bse.loc[pred]) if pred in bse.index else np.nan
            pval = float(pvalues.loc[pred]) if pred in pvalues.index else np.nan
            ci_lower = float(ci.loc[pred, 0]) if pred in ci.index else np.nan
            ci_upper = float(ci.loc[pred, 1]) if pred in ci.index else np.nan
            irr = float(np.exp(coef)) if np.isfinite(coef) else np.nan
            irr_ci_lower = float(np.exp(ci_lower)) if np.isfinite(ci_lower) else np.nan
            irr_ci_upper = float(np.exp(ci_upper)) if np.isfinite(ci_upper) else np.nan

            rows.append({
                'predictor': pred,
                'coef': coef,
                'bse': se,
                'pvalue': pval,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'IRR': irr,
                'IRR_ci_lower': irr_ci_lower,
                'IRR_ci_upper': irr_ci_upper
            })
        else:
            rows.append({
                'predictor': pred,
                'coef': np.nan,
                'bse': np.nan,
                'pvalue': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'IRR': np.nan,
                'IRR_ci_lower': np.nan,
                'IRR_ci_upper': np.nan
            })

    result_df = pd.DataFrame(rows).set_index('predictor')

    # Build a human-readable description summarizing significance and interpretation
    desc_lines = []
    # Note about model parameterization
    desc_lines.append(
        "Model: Negative binomial GLM with offset(log_seconds). Reported IRR = exp(coef) "
        "is the multiplicative effect on the expected rate of nuts opened per unit time."
    )

    # Mention whether clustered SEs were used
    cluster_note = ""
    if isinstance(model_output, dict) and 'glm_nb_clustered_result' in model_output:
        clustered_res = model_output['glm_nb_clustered_result']
        if hasattr(clustered_res, '_cluster_failure'):
            cluster_note = ("Note: attempted cluster-robust SEs by chimpanzee failed "
                            "during model fitting; results below use non-clustered inference.")
        else:
            cluster_note = ("Cluster-robust standard errors (clustered by chimpanzee) were used for inference.")
    else:
        cluster_note = "Clustered result not found; inference uses the provided fit object."
    desc_lines.append(cluster_note)

    # Interpret each predictor
    for pred in predictors:
        row = result_df.loc[pred]
        p = row['pvalue']
        irr = row['IRR']
        ci_lo = row['IRR_ci_lower']
        ci_hi = row['IRR_ci_upper']

        if np.isnan(row['coef']):
            desc_lines.append(f"{pred}: not present in the model results.")
            continue

        sig = False
        if not np.isnan(p):
            sig = (p < 0.05)

        # Friendly name and interpretation specifics
        if pred == 'age_c':
            name = "Age (mean-centered)"
            effect_unit = "per one-unit increase in age (mean-centered)"
        elif pred == 'sex_m':
            name = "Sex (male vs female)"
            effect_unit = "male relative to female"
        elif pred == 'help_y':
            name = "Received help (yes vs no)"
            effect_unit = "help received relative to no help"

        if np.isnan(irr):
            line = f"{name}: coefficient present but IRR could not be computed."
        else:
            sig_text = "statistically significant (p < 0.05)" if sig else "not statistically significant (p >= 0.05)"
            line = (
                f"{name}: IRR = {irr:.3f} (95% CI {ci_lo:.3f} to {ci_hi:.3f}), p = "
                f"{p:.3f} -> {sig_text}. Interpretation: {effect_unit} is associated with a "
                f"{'increase' if irr > 1 else 'decrease' if irr < 1 else 'no change'} "
                f"in the expected rate of nuts opened per unit time by a factor of {irr:.3f}."
            )
        desc_lines.append(line)

    description = "\n".join(desc_lines)

    return {
        "object": result_df,
        "description": description
    }