def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, p-values, 95% CIs for 'beauty' and 'beauty_sq',
    computes the marginal effect of beauty at the centered mean (beauty=0) and,
    if possible, at +/- 1 SD of beauty (if original data frame is accessible),
    and returns a concise interpretation.

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Text explanation of what these numbers mean re: impact of beauty"
      }
    """
    import math
    import numpy as np
    import pandas as pd

    # Basic extraction
    params = model_output.params
    bse = model_output.bse
    pvals = model_output.pvalues

    # Confidence intervals (array or DataFrame with rows in same order as params)
    ci = model_output.conf_int(alpha=0.05)
    # Determine parameter names robustly
    if hasattr(params, "index"):
        param_names = list(params.index)
    else:
        # fallback to model's exog names if params is array-like
        if hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
            param_names = list(model_output.model.exog_names)
        else:
            param_names = []

    def get_from_series_like(obj, idx, name):
        """Safely get value from a Series-like or ndarray-like object by name or index."""
        try:
            return float(obj[name])
        except Exception:
            try:
                return float(obj.iloc[idx])
            except Exception:
                try:
                    return float(np.asarray(obj)[idx])
                except Exception:
                    return float("nan")

    def get_ci_by_idx_or_name(idx, name):
        """Get CI lower and upper bounds handling DataFrame/ndarray forms."""
        # ci can be DataFrame or ndarray
        try:
            # If DataFrame with positional indexing
            low = float(ci.iloc[idx, 0])
            high = float(ci.iloc[idx, 1])
            return low, high
        except Exception:
            try:
                # If DataFrame with label-based indexing
                low = float(ci.loc[name].iloc[0])
                high = float(ci.loc[name].iloc[1])
                return low, high
            except Exception:
                try:
                    # If ndarray-like
                    arr = np.asarray(ci)
                    low = float(arr[idx, 0])
                    high = float(arr[idx, 1])
                    return low, high
                except Exception:
                    return float("nan"), float("nan")

    def get_param_info(name):
        if name not in param_names:
            return None
        idx = param_names.index(name)
        coef = get_from_series_like(params, idx, name)
        se_val = get_from_series_like(bse, idx, name)
        pval = get_from_series_like(pvals, idx, name)
        ci_low, ci_high = get_ci_by_idx_or_name(idx, name)
        return {
            "name": name,
            "coef": float(coef),
            "se": float(se_val),
            "pvalue": float(pval),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high)
        }

    beauty_info = get_param_info("beauty")
    beauty_sq_info = get_param_info("beauty_sq")

    # Covariance matrix for delta-method SEs (should be cluster-robust if model was fit that way)
    cov = model_output.cov_params()

    # Marginal effect of beauty at a given beauty value b: ME = beta_beauty + 2*beta_beauty_sq*b
    def marginal_effect_at(b):
        # handle missing params
        if beauty_info is None or beauty_sq_info is None:
            return None
        beta1 = beauty_info["coef"]
        beta2 = beauty_sq_info["coef"]
        me = beta1 + 2.0 * beta2 * b

        # Delta-method variance: Var(beta1 + 2*b*beta2) = Var(beta1) + (2*b)^2 Var(beta2) + 2*(2*b) Cov(beta1,beta2)
        try:
            # try label-based access first
            var_b1 = float(cov.loc["beauty", "beauty"])
            var_b2 = float(cov.loc["beauty_sq", "beauty_sq"])
            cov_b1b2 = float(cov.loc["beauty", "beauty_sq"])
            var_me = var_b1 + (2.0 * b) ** 2 * var_b2 + 2.0 * (2.0 * b) * cov_b1b2
            se_me = math.sqrt(max(var_me, 0.0))
            # 95% CI
            ci_low = me - 1.96 * se_me
            ci_high = me + 1.96 * se_me
        except Exception:
            try:
                # try positional access if cov is ndarray
                arr = np.asarray(cov)
                idx1 = param_names.index("beauty")
                idx2 = param_names.index("beauty_sq")
                var_b1 = float(arr[idx1, idx1])
                var_b2 = float(arr[idx2, idx2])
                cov_b1b2 = float(arr[idx1, idx2])
                var_me = var_b1 + (2.0 * b) ** 2 * var_b2 + 2.0 * (2.0 * b) * cov_b1b2
                se_me = math.sqrt(max(var_me, 0.0))
                ci_low = me - 1.96 * se_me
                ci_high = me + 1.96 * se_me
            except Exception:
                se_me = float("nan")
                ci_low = float("nan")
                ci_high = float("nan")

        return {"b": float(b), "marginal_effect": float(me), "se": se_me, "ci95_low": ci_low, "ci95_high": ci_high}

    # Marginal effect at mean beauty (data was centered, so mean = 0)
    me_at_mean = marginal_effect_at(0.0)

    # Try to compute +/- 1 SD of beauty if original dataframe is available
    me_at_plus_minus_1sd = {}
    sd_available = False
    try:
        # Try to access the original dataframe used to fit the model
        df = None
        if hasattr(model_output, "model") and hasattr(model_output.model, "data"):
            data_obj = model_output.model.data
            # Many statsmodels versions expose the frame at data.frame
            if hasattr(data_obj, "frame") and data_obj.frame is not None:
                df = data_obj.frame
            elif hasattr(data_obj, "orig_exog") and hasattr(data_obj, "orig_endog"):
                # fallback: attempt to reconstruct if possible (not guaranteed)
                try:
                    df = data_obj.frame
                except Exception:
                    df = None

        if df is not None and "beauty" in getattr(df, "columns", []):
            sd = float(df["beauty"].std(ddof=0))  # population SD consistent with centering
            sd_available = True
            me_at_plus = marginal_effect_at(sd)
            me_at_minus = marginal_effect_at(-sd)
            me_at_plus_minus_1sd = {"sd": sd, "plus_sd": me_at_plus, "minus_sd": me_at_minus}
    except Exception:
        sd_available = False

    # Construct a short conclusion about statistical significance / substantive effect
    conclusion_lines = []
    if beauty_info is not None:
        sig = "statistically significant" if beauty_info["pvalue"] < 0.05 else "not statistically significant"
        conclusion_lines.append(
            f"The linear (centered) beauty coefficient is {beauty_info['coef']:.4f} "
            f"(SE={beauty_info['se']:.4f}, p={beauty_info['pvalue']:.3f}), {sig} at alpha=0.05."
        )
        conclusion_lines.append(
            "Because beauty was centered at its mean, this linear coefficient represents\n"
            "the marginal effect of a one-unit increase in beauty for an instructor with\n"
            "average beauty (i.e., at the mean)."
        )
    else:
        conclusion_lines.append("The model does not contain a 'beauty' coefficient to report.")

    if beauty_sq_info is not None:
        sig2 = "statistically significant" if beauty_sq_info["pvalue"] < 0.05 else "not statistically significant"
        conclusion_lines.append(
            f"The quadratic term (beauty_sq) is {beauty_sq_info['coef']:.6f} "
            f"(SE={beauty_sq_info['se']:.6f}, p={beauty_sq_info['pvalue']:.3f}), {sig2}."
        )
        conclusion_lines.append(
            "A significant quadratic term would indicate nonlinearity in the beauty -> eval relationship."
        )

    # Interpret marginal effects
    if me_at_mean is not None:
        conclusion_lines.append(
            f"Marginal effect at mean beauty (beauty=0): {me_at_mean['marginal_effect']:.4f} "
            f"(SE={me_at_mean['se']:.4f}, 95% CI [{me_at_mean['ci95_low']:.4f}, {me_at_mean['ci95_high']:.4f}])."
        )

    if sd_available:
        conclusion_lines.append(
            f"Marginal effects at +/-1 SD (sd={me_at_plus_minus_1sd['sd']:.4f}) provided as well."
        )

    # Make a simple yes/no about whether beauty impacts evaluations:
    evidence = False
    reasons = []
    if beauty_info is not None and beauty_info["pvalue"] < 0.05:
        evidence = True
        reasons.append("linear term significant")
    if beauty_sq_info is not None and beauty_sq_info["pvalue"] < 0.05:
        evidence = True
        reasons.append("quadratic term significant")
    yes_no = "yes" if evidence else "no"
    conclusion_lines.append(
        f"Is there evidence that beauty impacts teaching evaluations? {yes_no}. "
        f"Reason(s): {', '.join(reasons) if reasons else 'neither coefficient is statistically significant at 0.05.'}"
    )

    # Assemble the object to return
    result_object = {
        "beauty": beauty_info,
        "beauty_sq": beauty_sq_info,
        "marginal_at_mean": me_at_mean,
        "marginal_at_plus_minus_1sd": me_at_plus_minus_1sd if sd_available else None,
        "sd_available": sd_available,
        "cov_params_used_for_delta_method": True
    }

    return {
        "object": result_object,
        "description": "\n".join(conclusion_lines)
    }