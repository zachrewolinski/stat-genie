def extract_final_answer(model_output):
    """
    Extracts statistics for the 'IsHuman' coefficient from a fitted statsmodels GLMResultsWrapper.
    Returns a dictionary with keys "object" and "description".

    "object" is a dictionary containing:
      - coef_log_odds: coefficient (log-odds) for IsHuman
      - se: standard error of the coefficient
      - z: z-statistic (coef / se)
      - p_value: p-value for the coefficient
      - ci_log_odds: 95% CI for the coefficient on log-odds scale [low, high]
      - odds_ratio: exp(coef)
      - odds_ratio_CI: 95% CI for odds ratio [low, high]
      - mean_pred_human: mean predicted AMTL_rate when IsHuman=1 for all records (weighted by AMTL_trials if present), else None
      - mean_pred_nonhuman: mean predicted AMTL_rate when IsHuman=0 for all records (weighted by AMTL_trials if present), else None
      - absolute_diff: mean_pred_human - mean_pred_nonhuman (or None)
      - relative_diff: (mean_pred_human - mean_pred_nonhuman) / mean_pred_nonhuman (or None)
      - conclusion: textual conclusion about whether humans have higher AMTL after adjustment (based on p<0.05)
    "description" is a short explanation of what the object contains and how to interpret it.
    """
    import numpy as np
    import pandas as pd

    res = model_output  # expected to be a statsmodels GLMResultsWrapper

    # Basic parameter extraction
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    conf = res.conf_int()  # DataFrame with [lower, upper]

    # Find parameter name containing 'IsHuman' (guard against unusual encoding)
    candidates = [n for n in params.index if 'IsHuman' in n]
    if not candidates:
        raise ValueError("Couldn't find a parameter name containing 'IsHuman' in model parameters: {}".format(list(params.index)))
    param_name = candidates[0]

    coef = float(params[param_name])
    se = float(bse[param_name])
    z = float(coef / se) if se != 0 else None
    p = float(pvalues[param_name])
    ci_low, ci_high = conf.loc[param_name].astype(float).values

    # Odds ratio and CI
    or_point = float(np.exp(coef))
    or_ci_low, or_ci_high = list(np.exp([ci_low, ci_high]).astype(float))

    # Attempt to compute mean predicted AMTL_rate for IsHuman=1 vs IsHuman=0
    mean_pred_h = None
    mean_pred_nh = None
    try:
        # Access original dataframe used to fit the model
        df_orig = res.model.data.frame.copy()

        # Ensure IsHuman column exists in that frame
        if 'IsHuman' not in df_orig.columns:
            raise KeyError("Original model data frame does not contain 'IsHuman' column")

        df_h = df_orig.copy()
        df_nh = df_orig.copy()
        df_h['IsHuman'] = 1
        df_nh['IsHuman'] = 0

        # Predictions: predict returns the mean response (probability) for GLM by default
        pred_h = res.predict(df_h)
        pred_nh = res.predict(df_nh)

        # If AMTL_trials present, use it as weights because observations represent groups of trials
        if 'AMTL_trials' in df_orig.columns:
            weights = df_orig['AMTL_trials'].astype(float)
            mean_pred_h = float(np.average(pred_h, weights=weights))
            mean_pred_nh = float(np.average(pred_nh, weights=weights))
        else:
            mean_pred_h = float(np.mean(pred_h))
            mean_pred_nh = float(np.mean(pred_nh))
    except Exception:
        # If anything fails, leave mean predictions as None (caller can inspect model_output separately)
        mean_pred_h = None
        mean_pred_nh = None

    absolute_diff = None if (mean_pred_h is None or mean_pred_nh is None) else float(mean_pred_h - mean_pred_nh)
    relative_diff = None
    if absolute_diff is not None and mean_pred_nh != 0:
        relative_diff = float(absolute_diff / mean_pred_nh)

    # Simple conclusion based on p-value and sign of coefficient
    alpha = 0.05
    if p < alpha:
        if coef > 0:
            conclusion = ("Statistically significant: modern humans (IsHuman=1) have higher AMTL frequency "
                          f"than non-human primates (p = {p:.3g}, coef (log-odds) = {coef:.4f}, OR = {or_point:.3f}).")
        else:
            conclusion = ("Statistically significant: modern humans (IsHuman=1) have lower AMTL frequency "
                          f"than non-human primates (p = {p:.3g}, coef (log-odds) = {coef:.4f}, OR = {or_point:.3f}).")
    else:
        conclusion = (f"No statistically significant difference in AMTL between modern humans and non-human primates "
                      f"(p = {p:.3g}, coef (log-odds) = {coef:.4f}, OR = {or_point:.3f}).")

    result_object = {
        "param_name": param_name,
        "coef_log_odds": coef,
        "se": se,
        "z": z,
        "p_value": p,
        "ci_log_odds": [float(ci_low), float(ci_high)],
        "odds_ratio": or_point,
        "odds_ratio_CI": [or_ci_low, or_ci_high],
        "mean_pred_human": mean_pred_h,
        "mean_pred_nonhuman": mean_pred_nh,
        "absolute_diff": absolute_diff,
        "relative_diff": relative_diff,
        "conclusion": conclusion
    }

    description = (
        "Extracted statistics for the model coefficient corresponding to 'IsHuman'. "
        "Coefficient is on the log-odds scale from the binomial (logit) GLM; odds_ratio = exp(coef). "
        "mean_pred_human / mean_pred_nonhuman are model-predicted AMTL rates obtained by setting IsHuman to 1 or 0 "
        "for all observations while leaving other covariates as observed; these are weighted by AMTL_trials if available. "
        "The 'conclusion' summarizes whether humans have higher AMTL after adjusting for age, sex, and tooth class "
        "based on a conventional p < 0.05 threshold."
    )

    return {"object": result_object, "description": description}