def extract_final_answer(model_output):
    """
    Extracts key statistics from a fitted statsmodels MixedLMResultsWrapper and
    returns a summary dict suitable for answering whether age, sex, and help
    influence nut-cracking efficiency.

    Returns:
      {
        "object": {
          "fixed_effects": {param_name: {estimate, se, z, pvalue, ci_lower, ci_upper, exp_estimate, exp_ci_lower, exp_ci_upper}, ...},
          "age_effects": {
             "age_no_help": {estimate, se, z, pvalue, ci_lower, ci_upper, exp_estimate, exp_ci_lower, exp_ci_upper},
             "age_with_help": {estimate, se, z, pvalue, ci_lower, ci_upper, exp_estimate, exp_ci_lower, exp_ci_upper}
          },
          "random_effects": { ... }  # variance / sd of random intercept if available
        },
        "description": "Plain-language interpretation of the results and significance"
      }
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Try to extract parameter table pieces in a robust way
    try:
        params = pd.Series(res.params)
    except Exception:
        # If params not available, fail early with helpful error
        raise ValueError("model_output has no 'params' attribute or it could not be read.")

    # Standard errors
    if hasattr(res, "bse"):
        bse = pd.Series(res.bse)
    else:
        bse = pd.Series(np.nan, index=params.index)

    # p-values (MixedLM usually provides pvalues)
    if hasattr(res, "pvalues"):
        pvalues = pd.Series(res.pvalues)
    else:
        pvalues = pd.Series(np.nan, index=params.index)

    # Confidence intervals (2.5% and 97.5%)
    try:
        ci = res.conf_int()  # DataFrame with two columns
        # Ensure it's a DataFrame indexed by param names
        if isinstance(ci, np.ndarray):
            # unlikely, but handle
            ci = pd.DataFrame(ci, index=params.index)
        ci.columns = ["ci_lower", "ci_upper"]
    except Exception:
        # Fallback to normal-approx CI from params and bse
        z_crit = 1.96
        ci = pd.DataFrame({
            "ci_lower": params - z_crit * bse,
            "ci_upper": params + z_crit * bse
        }, index=params.index)

    # Covariance matrix of parameter estimates (for combining coefficients)
    try:
        covp = res.cov_params()
        # If ndarray, convert to DataFrame with param names
        if not isinstance(covp, pd.DataFrame):
            covp = pd.DataFrame(covp, index=params.index, columns=params.index)
    except Exception:
        # If not available, create NaN matrix
        covp = pd.DataFrame(np.nan, index=params.index, columns=params.index)

    # Build fixed effects table
    fixed_effects = {}
    for name in params.index:
        est = float(params[name])
        se = float(bse.get(name, np.nan))
        pval = float(pvalues.get(name, np.nan)) if name in pvalues.index else np.nan
        ci_low = float(ci.loc[name, "ci_lower"]) if name in ci.index else np.nan
        ci_high = float(ci.loc[name, "ci_upper"]) if name in ci.index else np.nan
        # z-stat if possible
        z = est / se if (not np.isnan(se) and se != 0) else np.nan
        # exponentiated effect (multiplicative change in nuts/sec rate)
        exp_est = float(np.exp(est)) if not np.isnan(est) else np.nan
        exp_ci_low = float(np.exp(ci_low)) if not np.isnan(ci_low) else np.nan
        exp_ci_high = float(np.exp(ci_high)) if not np.isnan(ci_high) else np.nan

        fixed_effects[name] = {
            "estimate": est,
            "se": se,
            "z_or_t": z,
            "pvalue": pval,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "exp_estimate": exp_est,
            "exp_ci_lower": exp_ci_low,
            "exp_ci_upper": exp_ci_high
        }

    # Identify parameter names for the predictors of interest
    # age main effect likely named 'age_c'
    # sex main effect likely named 'sex_M'
    # help main effect likely named 'help_yes'
    # interaction could be 'age_c:help_yes' or 'age_c*help_yes' etc.
    def find_param_like(names, substr):
        for n in names:
            if substr == n:
                return n
        # fallback: substring match
        for n in names:
            if substr in n:
                return n
        return None

    names = list(params.index)
    age_name = find_param_like(names, "age_c")
    sex_name = find_param_like(names, "sex_M")
    help_name = find_param_like(names, "help_yes")
    # Find interaction containing both tokens
    interaction_name = None
    for n in names:
        if ("age_c" in n) and ("help_yes" in n):
            interaction_name = n
            break

    # Function to compute combined estimate (age when helped = age + interaction)
    def combine_linear(names_list):
        # names_list: list of param names to sum
        est = 0.0
        for n in names_list:
            est += float(params.get(n, 0.0))
        # variance of sum = sum variances + 2*sum covariances
        var = 0.0
        for i in range(len(names_list)):
            ni = names_list[i]
            vi = float(covp.loc[ni, ni]) if (ni in covp.index and ni in covp.columns) else np.nan
            if np.isnan(vi):
                var = np.nan
                break
            var += vi
            for j in range(i + 1, len(names_list)):
                nj = names_list[j]
                cov_ij = float(covp.loc[ni, nj]) if (ni in covp.index and nj in covp.columns) else np.nan
                if np.isnan(cov_ij):
                    var = np.nan
                    break
                var += 2.0 * cov_ij
            if np.isnan(var):
                break
        se = np.sqrt(var) if (not np.isnan(var) and var >= 0) else np.nan
        z = est / se if (not np.isnan(se) and se != 0) else np.nan
        # p-value for combined (normal approx)
        try:
            from scipy import stats
            pval = float(2.0 * stats.norm.sf(abs(z))) if (not np.isnan(z)) else np.nan
        except Exception:
            pval = np.nan
        ci_low = est - 1.96 * se if not np.isnan(se) else np.nan
        ci_high = est + 1.96 * se if not np.isnan(se) else np.nan
        exp_est = float(np.exp(est)) if not np.isnan(est) else np.nan
        exp_ci_low = float(np.exp(ci_low)) if not np.isnan(ci_low) else np.nan
        exp_ci_high = float(np.exp(ci_high)) if not np.isnan(ci_high) else np.nan

        return {
            "estimate": est,
            "se": se,
            "z_or_t": z,
            "pvalue": pval,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "exp_estimate": exp_est,
            "exp_ci_lower": exp_ci_low,
            "exp_ci_upper": exp_ci_high
        }

    age_effects = {}
    # Age effect when not helped (help_yes = 0) is simply the age coefficient
    if age_name is not None:
        age_effects["age_no_help"] = combine_linear([age_name])
    else:
        age_effects["age_no_help"] = None

    # Age effect when helped (help_yes = 1) is age + interaction
    if age_name is not None and interaction_name is not None:
        age_effects["age_with_help"] = combine_linear([age_name, interaction_name])
    else:
        # If no interaction parameter in model, it's the same as age_no_help
        age_effects["age_with_help"] = age_effects["age_no_help"]

    # Random effects summary: try to extract random intercept variance or SD
    random_effects = {}
    try:
        # cov_re is covariance matrix of random effects
        if hasattr(res, "cov_re"):
            cov_re = res.cov_re
            # If it's array-like, pick first diagonal as variance of random intercept (if only intercept)
            if isinstance(cov_re, np.ndarray):
                var_re = float(cov_re[0, 0]) if cov_re.size else np.nan
            else:
                # DataFrame or similar
                var_re = float(np.array(cov_re)[0, 0])
            random_effects["random_intercept_variance"] = var_re
            random_effects["random_intercept_sd"] = float(np.sqrt(var_re)) if (not np.isnan(var_re)) else np.nan
        else:
            # Try to read from res.cov_re if present
            if hasattr(res, "cov_re"):
                cre = res.cov_re
                random_effects["random_intercept_variance"] = float(cre[0, 0])
                random_effects["random_intercept_sd"] = float(np.sqrt(cre[0, 0]))
            else:
                random_effects["note"] = "No cov_re attribute found"
    except Exception:
        random_effects["note"] = "Could not extract random effects variance"

    # Build a plain-language description synthesizing significance
    def sig_label(p):
        if p is None or np.isnan(p):
            return "p=?, (not available)"
        if p < 0.001:
            return f"p<{0.001:.3f}"
        return f"p={p:.3f}"

    desc_lines = []
    desc_lines.append("Extracted fixed-effect estimates (on log(nuts/sec)):")
    # Summarize age, sex, help, and interaction if present
    for var in [age_name, sex_name, help_name, interaction_name]:
        if var is None:
            continue
        fe = fixed_effects[var]
        # Interpret: for raw coeff -> multiplicative effect = exp(coeff)
        desc_lines.append(
            f"- {var}: estimate={fe['estimate']:.4f}, se={fe['se']:.4f}, {sig_label(fe['pvalue'])}, "
            f"95% CI [{fe['ci_lower']:.4f}, {fe['ci_upper']:.4f}]. "
            f"Multiplicative change in rate: {fe['exp_estimate']:.3f} (95% CI [{fe['exp_ci_lower']:.3f}, {fe['exp_ci_upper']:.3f}])."
        )

    # Age effects summary
    if age_effects["age_no_help"] is not None:
        an = age_effects["age_no_help"]
        desc_lines.append(
            f"- Age effect when not helped: estimate={an['estimate']:.4f}, se={an['se']:.4f}, {sig_label(an['pvalue'])}. "
            f"Multiplicative change per year: {an['exp_estimate']:.3f} (95% CI [{an['exp_ci_lower']:.3f}, {an['exp_ci_upper']:.3f}])."
        )
    if age_effects["age_with_help"] is not None:
        ah = age_effects["age_with_help"]
        desc_lines.append(
            f"- Age effect when helped: estimate={ah['estimate']:.4f}, se={ah['se']:.4f}, {sig_label(ah['pvalue'])}. "
            f"Multiplicative change per year: {ah['exp_estimate']:.3f} (95% CI [{ah['exp_ci_lower']:.3f}, {ah['exp_ci_upper']:.3f}])."
        )

    # Add random effects note
    if "random_intercept_sd" in random_effects:
        desc_lines.append(
            f"Random intercept SD (chimpanzee): {random_effects['random_intercept_sd']:.4f} (variance={random_effects['random_intercept_variance']:.4f})."
        )
    else:
        desc_lines.append("Random intercept variance/SD not available.")

    # Short conclusion about influence: check significance of sex, help, and age (using main effects and interaction)
    concl = []
    alpha = 0.05
    # sex
    if sex_name in fixed_effects:
        p = fixed_effects[sex_name]["pvalue"]
        if (not np.isnan(p)) and p < alpha:
            concl.append("Sex (male vs female) appears to have a statistically significant effect on nut-cracking efficiency.")
        else:
            concl.append("No statistically significant effect of sex (male vs female) was detected.")
    # help
    if help_name in fixed_effects:
        p = fixed_effects[help_name]["pvalue"]
        if (not np.isnan(p)) and p < alpha:
            concl.append("Receiving help has a statistically significant main effect on efficiency (for an average/centered age).")
        else:
            concl.append("No statistically significant main effect of receiving help was detected (for average/centered age).")
    # age: consider interaction: if interaction significant then age effect depends on help
    age_sig = False
    if age_name in fixed_effects:
        p_age = fixed_effects[age_name]["pvalue"]
        p_int = fixed_effects[interaction_name]["pvalue"] if interaction_name in fixed_effects else np.nan
        if (not np.isnan(p_int)) and p_int < alpha:
            concl.append("The interaction between age and receiving help is statistically significant, so the effect of age differs depending on whether the individual received help.")
        else:
            # if interaction not significant, check age main effect
            if (not np.isnan(p_age)) and p_age < alpha:
                concl.append("Age has a statistically significant effect on efficiency (same effect regardless of help, since interaction is not significant).")
            else:
                concl.append("No statistically significant effect of age was detected.")
    # Combine
    desc_lines.append("Conclusion summary:")
    desc_lines.extend(["- " + c for c in concl])

    description = "\n".join(desc_lines)

    return {
        "object": {
            "fixed_effects": fixed_effects,
            "age_effects": age_effects,
            "random_effects": random_effects
        },
        "description": description
    }