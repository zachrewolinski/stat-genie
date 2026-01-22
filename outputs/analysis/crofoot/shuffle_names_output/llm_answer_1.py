def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLM (possibly robust) results object
    for the predictors of interest:
      - size_ratio_z (relative group size, standardized)
      - location_advantage_z (contest location advantage, standardized)
      - interaction of size_ratio_z and location_advantage_z
      - male_diff (control)

    Returns a dict with:
      - "object": dict mapping term -> stats (coef, se, z, p, 95% CI, odds ratio and CI, significant boolean)
      - "description": human-readable summary interpretation in the context of the task
    """
    import numpy as np

    # Attempt to access common result attributes (works for standard and robust results)
    try:
        params = model_output.params
        bse = model_output.bse
        # conf_int() usually returns a DataFrame or ndarray with two columns [lower, upper]
        conf = model_output.conf_int()
        # pvalues usually exists for statsmodels results (robust results should have it too)
        pvalues = getattr(model_output, "pvalues", None)
    except Exception as e:
        raise ValueError(f"Provided model_output does not look like a statsmodels results object: {e}")

    # If p-values are not present, compute them using normal approximation
    if pvalues is None:
        try:
            from scipy.stats import norm
            zvals = params / bse
            pvalues = 2 * (1 - norm.cdf(np.abs(zvals)))
        except Exception:
            # fallback: set pvalues to NaN
            pvalues = np.full_like(params, np.nan, dtype=float)
            pvalues = type(params)(pvalues)
            pvalues.index = params.index

    # Helper to find parameter name in params.index
    def find_param_name(target):
        # target can be 'a' or 'a:b' meaning both tokens must appear in param name (order-insensitive)
        if ":" in target:
            tokens = target.split(":")
            for name in params.index:
                if all(tok in name for tok in tokens):
                    return name
        else:
            # exact match preferred, else substring match
            if target in params.index:
                return target
            for name in params.index:
                if target in name:
                    return name
        return None

    # Define targets
    targets = {
        "size_ratio_z": "size_ratio_z",
        "location_advantage_z": "location_advantage_z",
        "interaction": "size_ratio_z:location_advantage_z",
        "male_diff": "male_diff"
    }

    results = {}
    for key, target in targets.items():
        pname = find_param_name(target)
        if pname is None:
            results[key] = {
                "found": False,
                "note": f"No parameter matching '{target}' was found in the model."
            }
            continue

        coef = float(params[pname])
        se = float(bse[pname]) if pname in bse.index else float(np.nan)
        zval = coef / se if se != 0 else float("nan")
        pval = float(pvalues[pname]) if pname in pvalues.index else float("nan")
        # confidence interval
        try:
            # conf may be a DataFrame or ndarray; rows correspond to param order
            if hasattr(conf, "loc"):  # DataFrame-like
                ci_low, ci_high = float(conf.loc[pname].iloc[0]), float(conf.loc[pname].iloc[1])
            else:
                # conf is ndarray; map param name to its position
                idx = list(params.index).index(pname)
                ci_low, ci_high = float(conf[idx, 0]), float(conf[idx, 1])
        except Exception:
            ci_low, ci_high = float("nan"), float("nan")

        # Odds ratio and CI (for logistic model)
        try:
            or_val = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low)) if not np.isnan(ci_low) else float("nan")
            or_ci_high = float(np.exp(ci_high)) if not np.isnan(ci_high) else float("nan")
        except Exception:
            or_val = or_ci_low = or_ci_high = float("nan")

        significant = (not np.isnan(pval)) and (pval < 0.05)

        results[key] = {
            "found": True,
            "param_name": pname,
            "coef": coef,
            "se": se,
            "z": zval,
            "p_value": pval,
            "ci_95": [ci_low, ci_high],
            "odds_ratio": or_val,
            "odds_ratio_ci_95": [or_ci_low, or_ci_high],
            "significant_0.05": bool(significant)
        }

    # Build a concise textual interpretation
    lines = []
    # summary for size_ratio_z
    if results["size_ratio_z"].get("found", False):
        r = results["size_ratio_z"]
        sig_text = "statistically significant (p < 0.05)" if r["significant_0.05"] else "not statistically significant (p >= 0.05)"
        lines.append(
            f"Relative group size (size_ratio_z) — coef={r['coef']:.3f}, OR={r['odds_ratio']:.3f}, "
            f"95% CI OR=[{r['odds_ratio_ci_95'][0]:.3f}, {r['odds_ratio_ci_95'][1]:.3f}], p={r['p_value']:.3g}; {sig_text}."
        )
    else:
        lines.append(results["size_ratio_z"]["note"])

    # summary for location_advantage_z
    if results["location_advantage_z"].get("found", False):
        r = results["location_advantage_z"]
        sig_text = "statistically significant (p < 0.05)" if r["significant_0.05"] else "not statistically significant (p >= 0.05)"
        lines.append(
            f"Contest location (location_advantage_z) — coef={r['coef']:.3f}, OR={r['odds_ratio']:.3f}, "
            f"95% CI OR=[{r['odds_ratio_ci_95'][0]:.3f}, {r['odds_ratio_ci_95'][1]:.3f}], p={r['p_value']:.3g}; {sig_text}."
        )
    else:
        lines.append(results["location_advantage_z"]["note"])

    # summary for interaction
    if results["interaction"].get("found", False):
        r = results["interaction"]
        sig_text = "statistically significant (p < 0.05)" if r["significant_0.05"] else "not statistically significant (p >= 0.05)"
        lines.append(
            f"Interaction (size_ratio_z x location_advantage_z) — coef={r['coef']:.3f}, p={r['p_value']:.3g}; {sig_text}. "
            "A significant interaction would indicate that the effect of relative group size on winning depends on contest location."
        )
    else:
        lines.append(results["interaction"]["note"])

    # male_diff
    if results["male_diff"].get("found", False):
        r = results["male_diff"]
        sig_text = "statistically significant (p < 0.05)" if r["significant_0.05"] else "not statistically significant (p >= 0.05)"
        lines.append(
            f"Control male_diff — coef={r['coef']:.3f}, OR={r['odds_ratio']:.3f}, p={r['p_value']:.3g}; {sig_text}."
        )
    else:
        lines.append(results["male_diff"]["note"])

    # Overall interpretation about the research question
    # If interaction significant, emphasize interaction; else comment on main effects
    interaction_sig = results.get("interaction", {}).get("significant_0.05", False)
    size_sig = results.get("size_ratio_z", {}).get("significant_0.05", False)
    loc_sig = results.get("location_advantage_z", {}).get("significant_0.05", False)

    if interaction_sig:
        lines.append(
            "Because the interaction term is significant, the influence of relative group size on the probability of winning "
            "depends on contest location. Interpret main-effect coefficients with caution; consider plotting predicted probabilities "
            "across combinations of size_ratio_z and location_advantage_z to visualize the interaction."
        )
    else:
        # No interaction: interpret main effects
        if size_sig and not loc_sig:
            lines.append("Conclusion: Relative group size is associated with focal-group victory (larger focal groups have higher odds), "
                         "while contest location shows no clear independent effect.")
        elif loc_sig and not size_sig:
            lines.append("Conclusion: Contest location (closer to focal home-range) is associated with focal-group victory, "
                         "while relative group size shows no clear independent effect.")
        elif size_sig and loc_sig:
            lines.append("Conclusion: Both relative group size and contest location independently predict focal-group victory.")
        else:
            lines.append("Conclusion: No strong evidence that relative group size or contest location (as modeled) independently predict focal-group victory.")

    description = "\n".join(lines)

    return {"object": results, "description": description}