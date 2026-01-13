def extract_final_answer(model_output):
    """
    Extract key statistics about the 'female' effect from the model output and
    produce a concise interpretation.

    Returns a dict with keys:
      - "object": a dictionary of numeric results (coeff, SE, p-values, CIs, AME)
      - "description": a short human-readable interpretation of the results
    """
    import numpy as np
    import pandas as pd

    out = {
        "object": None,
        "description": "Could not extract results (model object missing)."
    }

    if not isinstance(model_output, dict):
        out["description"] = "model_output must be a dict containing 'model_result' and/or 'average_marginal_effects'."
        return out

    model_result = model_output.get("model_result", None)
    marg_summary = model_output.get("average_marginal_effects", None)

    # Prepare containers for extracted values; allow missing values if some pieces not available
    coef = se = pval = ci_low = ci_high = None
    ame = ame_se = ame_p = ame_ci_low = ame_ci_high = None

    # Extract from the fitted model (logit) if available
    if model_result is not None:
        try:
            # coefficient (log-odds), standard error, p-value, and 95% CI from the fitted model
            params = getattr(model_result, "params", None)
            pvalues = getattr(model_result, "pvalues", None)
            conf = None
            try:
                conf = model_result.conf_int()
            except Exception:
                # some wrappers name method differently; try get_conf_int if present
                try:
                    conf = model_result.get_conf_int()
                except Exception:
                    conf = None

            if params is not None and "female" in params.index:
                coef = float(params.loc["female"])
            if pvalues is not None and "female" in pvalues.index:
                pval = float(pvalues.loc["female"])
            if conf is not None and "female" in conf.index:
                # conf may have columns [0,1] or named; take first two columns
                try:
                    ci_vals = conf.loc["female"].values
                    if len(ci_vals) >= 2:
                        ci_low, ci_high = float(ci_vals[0]), float(ci_vals[1])
                except Exception:
                    pass
        except Exception:
            # leave values as None if any extraction fails
            pass

    # Extract average marginal effect if available
    if marg_summary is not None:
        try:
            # marg_summary is expected to be a DataFrame with index containing 'female'
            ms = marg_summary
            if "female" in ms.index:
                row = ms.loc["female"]
                # dy/dx column commonly named 'dy/dx'
                if "dy/dx" in row.index:
                    ame = float(row["dy/dx"])
                else:
                    # try first numeric column as fallback
                    numeric_cols = [c for c in ms.columns if np.issubdtype(ms[c].dtype, np.number)]
                    if len(numeric_cols) >= 1:
                        ame = float(row[numeric_cols[0]])

                # standard error often 'Std. Err.' or 'Std. Err'
                se_candidates = ["Std. Err.", "Std. Err", "Std Err", "Std. Error", "std_err", "std"]
                for cand in se_candidates:
                    if cand in row.index:
                        ame_se = float(row[cand])
                        break
                # p-value candidates
                p_candidates = ["P>|z|", "P>|z| ", "pvalue", "p_value", "P>|z|", "P>|z|"]
                for cand in row.index:
                    if cand.lower().replace(" ", "") in ("p>|z|".replace(" ", ""), "pvalue", "p"):
                        try:
                            ame_p = float(row[cand])
                            break
                        except Exception:
                            pass
                # confidence interval column names may vary, try common ones
                ci_low_candidates = ["[0.025", "0.025", "Conf. Int. Low", "ci_lower", "lower", "0.025"]
                ci_high_candidates = ["0.975]", "0.975", "Conf. Int. Hi.", "ci_upper", "upper", "0.975"]
                # more robust: if columns contain 'Conf' or 'Int' try those
                for c in row.index:
                    if ("Conf" in str(c) and "Low" in str(c)) or str(c).strip().lower().startswith("0.025"):
                        try:
                            ame_ci_low = float(row[c])
                        except Exception:
                            pass
                    if ("Conf" in str(c) and ("Hi" in str(c) or "High" in str(c))) or str(c).strip().lower().startswith("0.975"):
                        try:
                            ame_ci_high = float(row[c])
                        except Exception:
                            pass
                # If not found using name heuristics, fallback to first/last numeric columns
                if ame_ci_low is None or ame_ci_high is None:
                    numeric_cols = [c for c in ms.columns if np.issubdtype(ms[c].dtype, np.number)]
                    if len(numeric_cols) >= 3:
                        # assume ordering dy/dx, Std. Err., z, P>|z|, [0.025, 0.975]
                        # pick last two numeric columns as CI if they look like CIs
                        possible_low = ms.loc["female", numeric_cols[-2]]
                        possible_high = ms.loc["female", numeric_cols[-1]]
                        try:
                            ame_ci_low = float(possible_low)
                            ame_ci_high = float(possible_high)
                        except Exception:
                            pass
        except Exception:
            pass

    # Decide which p-value to use for significance judgment: prefer AME p-value if available, else model coefficient p-value
    p_used = None
    if ame_p is not None:
        p_used = ame_p
    elif pval is not None:
        p_used = pval

    # Build a numeric summary dictionary to return under "object"
    numeric_summary = {
        "logit_coefficient_female": coef,
        "logit_std_err_female": se,
        "logit_pvalue_female": pval,
        "logit_ci_2.5_female": ci_low,
        "logit_ci_97.5_female": ci_high,
        "average_marginal_effect_female": ame,
        "ame_std_err_female": ame_se,
        "ame_pvalue_female": ame_p,
        "ame_ci_2.5_female": ame_ci_low,
        "ame_ci_97.5_female": ame_ci_high,
    }

    # Construct human-readable description
    desc_parts = []
    desc_parts.append("Effect of being female on probability of mortgage acceptance:")

    if ame is not None:
        # interpret AME on probability scale
        ame_pct = ame * 100.0
        desc_parts.append(
            f"- Average marginal effect (probability scale): {ame:.4f} "
            f"({ame_pct:.2f} percentage points)."
        )
        if ame_ci_low is not None and ame_ci_high is not None:
            desc_parts.append(f"  95% CI for AME: [{ame_ci_low:.4f}, {ame_ci_high:.4f}].")
        if ame_se is not None:
            desc_parts.append(f"  SE: {ame_se:.4f}.")
        if ame_p is not None:
            desc_parts.append(f"  p-value (AME): {ame_p:.4f}.")
    elif coef is not None:
        # fallback to log-odds interpretation
        desc_parts.append(f"- Log-odds coefficient from logit: {coef:.4f}.")
        if ci_low is not None and ci_high is not None:
            desc_parts.append(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}].")
        if pval is not None:
            desc_parts.append(f"  p-value: {pval:.4f}.")
        desc_parts.append("  (No marginal effect summary available.)")
    else:
        desc_parts.append("- No numeric results available for 'female'.")

    # Conclude about statistical significance at 5% level
    if p_used is not None:
        sig = p_used < 0.05
        if sig:
            desc_parts.append(f"- Conclusion: The effect is statistically significant (p = {p_used:.4f} < 0.05).")
        else:
            desc_parts.append(f"- Conclusion: The effect is NOT statistically significant (p = {p_used:.4f} >= 0.05).")
    else:
        desc_parts.append("- Conclusion: Could not determine statistical significance (p-value unavailable).")

    # Final plain-language conclusion about whether gender affects acceptance
    if ame is not None and p_used is not None:
        if p_used < 0.05:
            if ame < 0:
                desc_parts.append(f"  Plain language: Being female is associated with a lower probability of approval (~{abs(ame*100):.2f} percentage points lower).")
            else:
                desc_parts.append(f"  Plain language: Being female is associated with a higher probability of approval (~{abs(ame*100):.2f} percentage points higher).")
        else:
            desc_parts.append("  Plain language: The estimated difference (~{:.2f} percentage points) is not statistically different from zero; we cannot conclude gender affects approval.".format(ame*100))
    elif coef is not None and p_used is not None:
        # Use log-odds magnitude for a cautious plain-language remark
        if p_used < 0.05:
            direction = "lower" if coef < 0 else "higher"
            desc_parts.append(f"  Plain language: Being female is associated with {direction} odds of approval (log-odds = {coef:.4f}).")
        else:
            desc_parts.append("  Plain language: No statistically significant effect of gender on approval was detected.")

    out["object"] = numeric_summary
    out["description"] = " ".join(desc_parts)

    return out