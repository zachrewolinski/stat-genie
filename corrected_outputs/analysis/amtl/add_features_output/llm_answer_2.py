def extract_final_answer(model_output):
    """
    Extract statistics related to the effect of being a modern human (IsHuman)
    from a fitted statsmodels GLMResults/GLMResultsWrapper object.

    Returns a dictionary with keys:
      - "object": dict of numeric results (coef, se, z, p, conf_int, odds_ratio, odds_ratio_conf_int,
                  predicted_prob_diff (if computable), and the exact parameter name used)
      - "description": a short plain-language interpretation answering whether modern humans
                       have higher AMTL after accounting for covariates, based on the sign
                       and statistical significance of the IsHuman effect.

    The function is robust to the exact parameter name used for IsHuman (e.g., "IsHuman",
    "C(IsHuman)[T.1]" etc.) by searching for any parameter name that contains the substring
    "IsHuman".
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Prepare a helper to fail gracefully
    result_obj = {
        'param_name': None,
        'coef': None,
        'se': None,
        'z': None,
        'p_value': None,
        'conf_int_95': None,
        'odds_ratio': None,
        'odds_ratio_conf_int_95': None,
        'predicted_prob_Human_minus_NonHuman': None  # estimated change in predicted probability at mean covariates if computable
    }

    # Identify parameter name for IsHuman (allowing categorical naming)
    try:
        params = res.params  # pandas Series
    except Exception as e:
        raise ValueError(f"Unable to access model parameters from model_output: {e}")

    # Find candidate parameter names that refer to IsHuman
    candidates = [name for name in params.index if 'IsHuman' in str(name)]
    if len(candidates) == 0:
        raise ValueError("Could not find any parameter name containing 'IsHuman' in model parameters: "
                         f"found parameters {list(params.index)}")

    # Prefer exact match 'IsHuman' if present; otherwise take the first candidate
    if 'IsHuman' in candidates:
        pname = 'IsHuman'
    else:
        pname = candidates[0]

    result_obj['param_name'] = pname

    # Extract coef, se, p-value, conf-int
    try:
        coef = float(params[pname])
        se = float(res.bse[pname])
        # z-score (Wald) for GLM with logit link
        z = coef / se if se != 0 else np.nan
        pval = float(res.pvalues[pname])
        ci_df = res.conf_int()
        if pname in ci_df.index:
            ci_lower, ci_upper = float(ci_df.loc[pname, 0]), float(ci_df.loc[pname, 1])
        else:
            # if conf_int rows are not indexed by param names for some reason
            ci_lower, ci_upper = (np.nan, np.nan)

        # Odds ratio and CI
        or_est = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else np.nan
        or_ci_upper = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else np.nan

        result_obj.update({
            'coef': coef,
            'se': se,
            'z': z,
            'p_value': pval,
            'conf_int_95': (ci_lower, ci_upper),
            'odds_ratio': or_est,
            'odds_ratio_conf_int_95': (or_ci_lower, or_ci_upper)
        })
    except Exception as e:
        raise ValueError(f"Error extracting coefficient information for parameter '{pname}': {e}")

    # Attempt to estimate the difference in predicted AMTL probability for a hypothetical
    # individual who is human vs non-human, holding other covariates at their dataset means
    # and using a reference level for tooth_class. This requires access to the original DataFrame.
    try:
        # statsmodels stores the original DataFrame (when using formulas) at model.data.frame in many versions
        df_original = None
        if hasattr(res, 'model') and hasattr(res.model, 'data') and hasattr(res.model.data, 'frame'):
            df_original = res.model.data.frame
        elif hasattr(res, 'model') and hasattr(res.model, 'data'):
            # fallback: some versions have different attribute names
            df_original = getattr(res.model.data, 'orig_endog', None)

        if isinstance(df_original, pd.DataFrame):
            # compute means for numeric covariates used in the formula
            mean_age = float(df_original['age'].mean()) if 'age' in df_original.columns else 0.0
            mean_probmale = float(df_original['ProbMale'].mean()) if 'ProbMale' in df_original.columns else 0.0

            # choose a tooth_class level for prediction:
            if 'tooth_class' in df_original.columns:
                # use the most frequent observed level as a reasonable reference
                ref_tooth = df_original['tooth_class'].mode().iloc[0]
            else:
                ref_tooth = None

            # build two rows (IsHuman = 0 vs 1)
            rows = []
            for ishuman in [0, 1]:
                row = {}
                row['IsHuman'] = ishuman
                row['age'] = mean_age
                row['ProbMale'] = mean_probmale
                if ref_tooth is not None:
                    row['tooth_class'] = ref_tooth
                rows.append(row)
            newdf = pd.DataFrame(rows)

            # use the model's predict method; statsmodels will handle categorical expansion if formula used
            preds = res.predict(newdf)  # yields predicted mean response (probability for Binomial)
            # In case predict returns an array-like
            p_nonhuman = float(preds[0])
            p_human = float(preds[1])
            prob_diff = p_human - p_nonhuman

            result_obj['predicted_prob_Human_minus_NonHuman'] = prob_diff
        else:
            # Could not access original data; leave predicted difference as None
            result_obj['predicted_prob_Human_minus_NonHuman'] = None
    except Exception:
        # If prediction fails for any reason, do not crash—leave as None
        result_obj['predicted_prob_Human_minus_NonHuman'] = None

    # Formulate a concise interpretation / answer to the Yes/No question
    coef_sign = np.sign(result_obj['coef']) if result_obj['coef'] is not None else 0
    pval = result_obj['p_value']

    if result_obj['coef'] is None or pval is None:
        conclusion_text = ("Could not extract the IsHuman effect from the model output; "
                           "no conclusion can be drawn programmatically.")
    else:
        if coef_sign > 0 and pval < 0.05:
            conclusion_text = (
                "Yes: the estimated coefficient for IsHuman is positive and statistically significant "
                f"(coef = {result_obj['coef']:.4f}, p = {pval:.3g}). This indicates that, after "
                "controlling for age, ProbMale, and tooth class, modern humans have higher odds of "
                "antemortem tooth loss than the non-human primates in the dataset. "
                f"Odds ratio = {result_obj['odds_ratio']:.3f} (95% CI {result_obj['odds_ratio_conf_int_95'][0]:.3f}, "
                f"{result_obj['odds_ratio_conf_int_95'][1]:.3f})."
            )
        elif coef_sign > 0 and pval >= 0.05:
            conclusion_text = (
                "No strong evidence: the point estimate for IsHuman is positive (suggesting higher AMTL in humans), "
                f"but it is not statistically significant (coef = {result_obj['coef']:.4f}, p = {pval:.3g}). "
                "Thus we cannot confidently conclude that modern humans have higher AMTL after adjustment."
            )
        elif coef_sign < 0 and pval < 0.05:
            conclusion_text = (
                "No (inverse): the estimated coefficient for IsHuman is negative and statistically significant "
                f"(coef = {result_obj['coef']:.4f}, p = {pval:.3g}), indicating that modern humans have lower odds "
                "of AMTL compared to the non-human primates after adjustment. "
                f"Odds ratio = {result_obj['odds_ratio']:.3f} (95% CI {result_obj['odds_ratio_conf_int_95'][0]:.3f}, "
                f"{result_obj['odds_ratio_conf_int_95'][1]:.3f})."
            )
        else:
            conclusion_text = (
                "No evidence of a difference: the coefficient for IsHuman is negative (suggesting lower AMTL in humans) "
                f"but not statistically significant (coef = {result_obj['coef']:.4f}, p = {pval:.3g})."
            )

    # Return the result object and a human-readable description
    return {
        "object": result_obj,
        "description": conclusion_text
    }