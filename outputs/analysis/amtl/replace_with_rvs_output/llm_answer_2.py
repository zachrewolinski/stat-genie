def extract_final_answer(model_output):
    """
    Extracts statistics comparing non-human genera to the reference genus (Homo sapiens)
    from a fitted statsmodels GLMResultsWrapper (fitted with Treatment(reference="Homo sapiens")
    for genus). Returns a dict with:
      - "object": a pandas.DataFrame summarizing coef, SE, p-value, conf. intervals,
                  odds ratios and whether the result implies Homo sapiens has higher AMTL
                  (at alpha=0.05) for each non-human genus.
      - "description": a textual interpretation of those results in the context of the task.

    Notes:
    - The model is assumed to be a binomial GLM with logit link (so coefficients are
      log-odds). The function exponentiates coefficients and CIs to produce odds ratios.
    - It expects parameter names produced by Patsy/statsmodels such as
      'C(genus, Treatment(reference="Homo sapiens"))[T.Pan]' (but will try to match
      other common naming patterns that include 'genus' and 'T.').
    """
    import re
    import numpy as np
    import pandas as pd

    # Pull estimates, SEs, p-values, and confidence intervals from the fitted model output.
    try:
        params = model_output.params
        bse = model_output.bse
        pvalues = model_output.pvalues
        conf = model_output.conf_int()  # DataFrame or array with two columns (lower, upper)
    except Exception as e:
        raise ValueError("model_output does not appear to be a statsmodels results object with "
                         "params/bse/pvalues/conf_int attributes.") from e

    # Locate genus-related parameter rows and extract genus names
    genus_rows = []
    genus_names = []
    for name in params.index:
        if 'genus' in name:
            # try to extract the genus label after 'T.' (common pattern)
            m = re.search(r'T\.([^\]\s]+)', name)
            if m:
                g = m.group(1)
            else:
                # fallback: take the last token after '.' or '['
                if '.' in name:
                    g = name.split('.')[-1].strip(']').strip()
                else:
                    g = name
            genus_rows.append(name)
            genus_names.append(g)

    if len(genus_rows) == 0:
        # No genus coefficients found; return a helpful error object
        return {
            "object": None,
            "description": (
                "No genus-level coefficients were found in the provided model_output. "
                "Ensure the model was fitted with genus as a categorical predictor and that "
                "the reference level was set to 'Homo sapiens' (e.g., via "
                "C(genus, Treatment(reference='Homo sapiens'))). Parameter names in the model "
                "were: {}".format(list(params.index))
            )
        }

    # Build a summary table for the non-human genera
    records = []
    alpha = 0.05
    for row_name, genus in zip(genus_rows, genus_names):
        coef = float(params[row_name])
        se = float(bse[row_name]) if row_name in bse.index else np.nan
        pval = float(pvalues[row_name]) if row_name in pvalues.index else np.nan

        # conf may be DataFrame or ndarray; handle both
        try:
            ci_low, ci_high = conf.loc[row_name].values
        except Exception:
            try:
                idx = list(params.index).index(row_name)
                ci_low, ci_high = conf[idx, 0], conf[idx, 1]
            except Exception:
                ci_low, ci_high = (np.nan, np.nan)

        # Odds ratio interpretation (exp of log-odds coef)
        or_est = float(np.exp(coef)) if np.isfinite(coef) else np.nan
        or_ci_low = float(np.exp(ci_low)) if np.isfinite(ci_low) else np.nan
        or_ci_high = float(np.exp(ci_high)) if np.isfinite(ci_high) else np.nan

        # Since 'Homo sapiens' was the reference, a negative coefficient means the non-human genus
        # has LOWER log-odds (and thus lower odds/probability) of AMTL relative to Homo sapiens.
        # Therefore, Homo sapiens would have higher AMTL if coef < 0 and the difference is statistically significant.
        humans_higher = (coef < 0) and (pval < alpha)

        interpretation = ""
        if np.isnan(pval):
            interpretation = "No p-value available."
        else:
            if humans_higher:
                interpretation = (
                    "Homo sapiens shows significantly higher AMTL than {} "
                    "(coef={:.3g}, p={:.3g}; OR={:.3g}, 95% CI [{:.3g}, {:.3g}])."
                    .format(genus, coef, pval, or_est, or_ci_low, or_ci_high)
                )
            else:
                # either non-significant or in opposite direction
                if pval < alpha and coef > 0:
                    interpretation = (
                        "{} shows significantly higher AMTL than Homo sapiens "
                        "(coef={:.3g}, p={:.3g}; OR={:.3g}, 95% CI [{:.3g}, {:.3g}])."
                        .format(genus, coef, pval, or_est, or_ci_low, or_ci_high)
                    )
                else:
                    interpretation = (
                        "No statistically significant difference in AMTL between Homo sapiens and {} "
                        "(coef={:.3g}, p={:.3g}; OR={:.3g}, 95% CI [{:.3g}, {:.3g}])."
                        .format(genus, coef, pval, or_est, or_ci_low, or_ci_high)
                    )

        records.append({
            "genus_param": row_name,
            "genus": genus,
            "coef_log_odds": coef,
            "se": se,
            "p_value": pval,
            "ci95_low_logodds": ci_low,
            "ci95_high_logodds": ci_high,
            "odds_ratio": or_est,
            "or_ci95_low": or_ci_low,
            "or_ci95_high": or_ci_high,
            "humans_higher_at_0.05": bool(humans_higher),
            "interpretation": interpretation
        })

    summary_df = pd.DataFrame.from_records(records).set_index("genus")

    # Create an overall textual conclusion based on the genus-level results
    if summary_df["humans_higher_at_0.05"].all():
        overall = (
            "Overall conclusion: YES — after adjusting for age, prob_male, and tooth class, "
            "modern humans (Homo sapiens) have higher AMTL than all examined non-human genera "
            "(Pan, Pongo, Papio) at alpha = 0.05."
        )
    elif summary_df["humans_higher_at_0.05"].any():
        sig_genera = list(summary_df[summary_df["humans_higher_at_0.05"]].index)
        nonsig_genera = list(summary_df[~summary_df["humans_higher_at_0.05"]].index)
        overall = (
            "Overall conclusion: MIXED — Homo sapiens shows significantly higher AMTL than {} "
            "but not (significantly) different from {} after adjustment (alpha = 0.05)."
            .format(", ".join(sig_genera), ", ".join(nonsig_genera) if nonsig_genera else "none")
        )
    else:
        # No genus shows humans higher; check if any genus shows higher AMTL than humans
        any_nonhuman_higher = (summary_df["p_value"] < alpha) & (summary_df["coef_log_odds"] > 0)
        if any_nonhuman_higher.any():
            higher_genera = list(summary_df[any_nonhuman_higher].index)
            overall = (
                "Overall conclusion: NO — there is no evidence that Homo sapiens has higher AMTL. "
                "Some non-human genera ({}) have significantly higher AMTL than Homo sapiens."
                .format(", ".join(higher_genera))
            )
        else:
            overall = (
                "Overall conclusion: NO — there is no statistically significant evidence that modern humans "
                "have higher AMTL than the examined non-human genera after adjusting for age, prob_male, "
                "and tooth class (alpha = 0.05)."
            )

    description = (
        "This table shows, for each non-human genus, the model coefficient (log-odds difference vs. "
        "Homo sapiens), its standard error, p-value, 95% confidence interval (log-odds), and the odds "
        "ratio with 95% CI. Because the model used treatment coding with 'Homo sapiens' as the reference, "
        "a negative coefficient means the given non-human genus has lower AMTL than Homo sapiens; if that "
        "difference is statistically significant (p < 0.05) we conclude Homo sapiens has higher AMTL for that genus. "
        + "Alpha = 0.05 was used to determine significance. " + overall
    )

    return {
        "object": summary_df,
        "description": description
    }