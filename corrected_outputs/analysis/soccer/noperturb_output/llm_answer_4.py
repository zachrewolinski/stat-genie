def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, 95% CI, and exponentiated effect (IRR)
    for the 'SkinDark' variable from a statsmodels GLM/ResultsWrapper object.
    
    Returns a dictionary with keys:
      - "object": dict of numeric results (coef, se, pval, conf_int, IRR, IRR_CI)
      - "description": textual interpretation of the coefficient in the context of the task
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic sanity checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object with .params")

    params = res.params
    # Find the parameter corresponding to SkinDark (allow for possible name variants)
    skin_param_name = None
    for name in params.index:
        if "SkinDark" in str(name):
            skin_param_name = name
            break

    if skin_param_name is None:
        raise ValueError("Could not find a parameter with name containing 'SkinDark' in model params: "
                         f"{list(params.index)}")

    # Extract coefficient
    coef = float(params[skin_param_name])

    # Standard error: prefer res.bse if available, else compute from cov_params
    if hasattr(res, "bse"):
        se = float(res.bse[skin_param_name])
    else:
        cov = res.cov_params()
        se = float(np.sqrt(np.abs(cov.loc[skin_param_name, skin_param_name])))

    # p-value: prefer res.pvalues if available
    if hasattr(res, "pvalues"):
        pval = float(res.pvalues[skin_param_name])
    else:
        # compute z and two-sided p-value
        z = coef / se if se != 0 else np.nan
        from scipy import stats
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))

    # 95% confidence interval on the coefficient (log scale)
    if hasattr(res, "conf_int"):
        try:
            ci = res.conf_int().loc[skin_param_name].astype(float)
            ci_low, ci_high = float(ci.iloc[0]), float(ci.iloc[1])
        except Exception:
            # fallback if conf_int returns ndarray or different indexing
            ci_arr = res.conf_int()
            # try to find row index
            try:
                row_idx = list(res.params.index).index(skin_param_name)
                ci_low, ci_high = float(ci_arr[row_idx, 0]), float(ci_arr[row_idx, 1])
            except Exception:
                ci_low, ci_high = float(coef - 1.96 * se), float(coef + 1.96 * se)
    else:
        ci_low, ci_high = float(coef - 1.96 * se), float(coef + 1.96 * se)

    # Exponentiate to get incidence rate ratio (IRR) and its CI
    irr = float(np.exp(coef))
    irr_ci_low = float(np.exp(ci_low))
    irr_ci_high = float(np.exp(ci_high))

    # Build the returned object
    result_obj = {
        "param_name": str(skin_param_name),
        "coef_log": coef,
        "se": se,
        "p_value": pval,
        "conf_int_log": (ci_low, ci_high),
        "IRR": irr,
        "IRR_95CI": (irr_ci_low, irr_ci_high)
    }

    # Interpretation text
    # For a GLM negative binomial with log link and offset = log(games), coef is log rate ratio.
    signif = "statistically significant" if (not np.isnan(pval) and pval < 0.05) else "not statistically significant"
    direction = "higher" if irr > 1 else ("lower" if irr < 1 else "no difference")
    description = (
        f"The model coefficient for '{skin_param_name}' (dark skin vs light skin) is {coef:.4f} "
        f"(SE = {se:.4f}, p = {pval:.3g}), with a 95% CI on the log scale of "
        f"({ci_low:.4f}, {ci_high:.4f}). Exponentiating yields an incidence rate ratio (IRR) = {irr:.3f} "
        f"with 95% CI = ({irr_ci_low:.3f}, {irr_ci_high:.3f}).\n\n"
        f"Interpretation: in this dyad-level negative binomial model with log(games) as an offset, "
        f"an IRR > 1 means players coded as having a dark skin tone receive more red cards per game "
        f"compared to players coded as having a light skin tone. Here the IRR = {irr:.3f}, indicating "
        f"{'an increase' if irr>1 else 'a decrease' if irr<1 else 'no change'} in the expected red card rate "
        f"for dark-skinned players relative to light-skinned players. The effect is {signif} at the 5% level."
    )

    return {"object": result_obj, "description": description}