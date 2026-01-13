def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of ChildrenYes on extramarital affairs
    from the provided model_output dict (expected keys: 'logit' or 'logit_glm_binomial',
    and 'neg_binomial' or 'poisson' as fallbacks).

    Returns:
      {
        "object": {
          "logit": { ... } or None,
          "count": { ... } or None
        },
        "description": "Plain-language interpretation of the extracted stats."
      }
    The "logit" and "count" dicts contain coefficients, p-values, confidence intervals,
    transformed effect measures (odds ratios or incidence-rate ratios), and separate
    marginal effects for women (gender_male=0) and men (gender_male=1) when interaction
    term is present.
    """
    import numpy as np

    out = {"logit": None, "count": None}
    desc_lines = []

    # Helper to find a model result by possible keys
    def find_model(keys):
        for k in keys:
            if k in model_output and model_output[k] is not None:
                return model_output[k]
        return None

    # Locate logistic result (Logit or GLM binomial)
    logit_res = find_model(['logit', 'logit_glm_binomial'])
    # Locate count result (Negative binomial, otherwise Poisson)
    count_res = find_model(['neg_binomial', 'neg_binomial_res', 'poisson'])

    var = 'ChildrenYes'
    interaction = 'Children_gender_interaction'
    gender_var = 'gender_male'
    alpha = 0.05
    z = 1.96  # approx 95% CI

    if logit_res is not None:
        try:
            params = logit_res.params
            pvalues = logit_res.pvalues
            conf = logit_res.conf_int()
            cov = logit_res.cov_params()

            if var in params.index:
                coef = float(params[var])
                pval = float(pvalues[var]) if var in pvalues.index else None
                ci_low = float(conf.loc[var, 0])
                ci_high = float(conf.loc[var, 1])
                or_val = float(np.exp(coef))
                or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))

                logit_dict = {
                    "coef_children": coef,
                    "pvalue_children": pval,
                    "ci_children": (ci_low, ci_high),
                    "odds_ratio_children": or_val,
                    "odds_ratio_children_ci": or_ci
                }

                # If interaction and gender variable present, compute marginal effects for men and women
                if (interaction in params.index) and (gender_var in params.index):
                    coef_int = float(params[interaction])
                    # women's effect (gender_male = 0): coef
                    # men's effect (gender_male = 1): coef + coef_int
                    coef_men = coef + coef_int

                    # compute p-value for interaction if available
                    pval_int = float(pvalues[interaction]) if interaction in pvalues.index else None

                    # compute CI for men's combined coef using covariance matrix
                    try:
                        var_sum = cov.loc[var, var] + cov.loc[interaction, interaction] + 2 * cov.loc[var, interaction]
                        se_sum = float(np.sqrt(var_sum))
                        ci_men_low = coef_men - z * se_sum
                        ci_men_high = coef_men + z * se_sum
                    except Exception:
                        # fallback if cov not available
                        se_men = None
                        ci_men_low = None
                        ci_men_high = None

                    or_women = float(np.exp(coef))
                    or_women_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
                    or_men = float(np.exp(coef_men)) if coef_men is not None else None
                    or_men_ci = (float(np.exp(ci_men_low)), float(np.exp(ci_men_high))) if (ci_men_low is not None) else (None, None)

                    logit_dict.update({
                        "coef_interaction": coef_int,
                        "pvalue_interaction": pval_int,
                        "coef_children_men": coef_men,
                        "ci_children_men": (ci_men_low, ci_men_high),
                        "odds_ratio_children_women": or_women,
                        "odds_ratio_children_women_ci": or_women_ci,
                        "odds_ratio_children_men": or_men,
                        "odds_ratio_children_men_ci": or_men_ci
                    })
                out["logit"] = logit_dict

                # Add human-readable summary about statistical significance
                sig = (pval is not None) and (pval < alpha)
                sign = "decrease" if coef < 0 else ("increase" if coef > 0 else "no change")
                desc_lines.append(
                    "Logistic model (any affair): ChildrenYes coef = {:.4f}, p = {:.4g}. "
                    "This corresponds to an odds ratio = {:.3f} (95% CI [{:.3f}, {:.3f}]). "
                    "Coefficient sign suggests a {} in odds of any affair for the reference gender (gender_male=0).{}".format(
                        coef, pval if pval is not None else float('nan'),
                        or_val, or_ci[0], or_ci[1],
                        sign,
                        "" if not ((interaction in params.index) and (gender_var in params.index)) else
                        " A gender interaction is present; see male/female marginal effects below."
                    )
                )
                if (interaction in params.index) and (gender_var in params.index):
                    # describe men effect
                    coef_men = logit_dict.get("coef_children_men")
                    pval_int = logit_dict.get("pvalue_interaction")
                    or_men = logit_dict.get("odds_ratio_children_men")
                    ci_men = logit_dict.get("odds_ratio_children_men_ci")
                    # For men, statistical significance of combined effect is not directly available as a p-value here,
                    # but we can note significance of main and interaction terms.
                    desc_lines.append(
                        "For women (gender_male=0): OR = {:.3f} (95% CI [{:.3f}, {:.3f}]).".format(
                            logit_dict["odds_ratio_children_women"],
                            logit_dict["odds_ratio_children_women_ci"][0],
                            logit_dict["odds_ratio_children_women_ci"][1]
                        )
                    )
                    desc_lines.append(
                        "For men (gender_male=1): combined OR = {:.3f} (approx 95% CI [{:.3f}, {:.3f}]). "
                        "Interaction term p = {}.".format(
                            or_men,
                            ci_men[0] if ci_men[0] is not None else float('nan'),
                            ci_men[1] if ci_men[1] is not None else float('nan'),
                            pval_int if pval_int is not None else "NA"
                        )
                    )
            else:
                desc_lines.append("Logistic model present but variable '{}' not found in model.".format(var))
        except Exception as e:
            desc_lines.append("Error extracting from logistic model: {}".format(e))

    else:
        desc_lines.append("No logistic/binomial model found in model_output.")

    if count_res is not None:
        try:
            params = count_res.params
            pvalues = count_res.pvalues
            conf = count_res.conf_int()
            cov = count_res.cov_params()

            if var in params.index:
                coef = float(params[var])
                pval = float(pvalues[var]) if var in pvalues.index else None
                ci_low = float(conf.loc[var, 0])
                ci_high = float(conf.loc[var, 1])
                irr = float(np.exp(coef))
                irr_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))

                count_dict = {
                    "coef_children": coef,
                    "pvalue_children": pval,
                    "ci_children": (ci_low, ci_high),
                    "irr_children": irr,
                    "irr_children_ci": irr_ci,
                    "model_family": getattr(count_res, "model", None).family.__class__.__name__ if getattr(count_res, "model", None) is not None else None
                }

                # Interaction handling for count model
                if (interaction in params.index) and (gender_var in params.index):
                    coef_int = float(params[interaction])
                    coef_men = coef + coef_int
                    pval_int = float(pvalues[interaction]) if interaction in pvalues.index else None
                    try:
                        var_sum = cov.loc[var, var] + cov.loc[interaction, interaction] + 2 * cov.loc[var, interaction]
                        se_sum = float(np.sqrt(var_sum))
                        ci_men_low = coef_men - z * se_sum
                        ci_men_high = coef_men + z * se_sum
                    except Exception:
                        ci_men_low = None
                        ci_men_high = None

                    irr_women = float(np.exp(coef))
                    irr_women_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
                    irr_men = float(np.exp(coef_men)) if coef_men is not None else None
                    irr_men_ci = (float(np.exp(ci_men_low)), float(np.exp(ci_men_high))) if (ci_men_low is not None) else (None, None)

                    count_dict.update({
                        "coef_interaction": coef_int,
                        "pvalue_interaction": pval_int,
                        "coef_children_men": coef_men,
                        "ci_children_men": (ci_men_low, ci_men_high),
                        "irr_children_women": irr_women,
                        "irr_children_women_ci": irr_women_ci,
                        "irr_children_men": irr_men,
                        "irr_children_men_ci": irr_men_ci
                    })

                out["count"] = count_dict

                sig = (pval is not None) and (pval < alpha)
                sign = "decrease" if coef < 0 else ("increase" if coef > 0 else "no change")
                desc_lines.append(
                    "Count model (frequency of affairs, {}): ChildrenYes coef = {:.4f}, p = {:.4g}. "
                    "This corresponds to an IRR = {:.3f} (95% CI [{:.3f}, {:.3f}]). "
                    "Coefficient sign suggests a {} in the expected count of affairs for the reference gender.".format(
                        count_dict.get("model_family", "count model"),
                        coef, pval if pval is not None else float('nan'),
                        irr, irr_ci[0], irr_ci[1],
                        sign
                    )
                )
                if (interaction in params.index) and (gender_var in params.index):
                    desc_lines.append(
                        "See male/female marginal IRRs in the 'object' field returned."
                    )
            else:
                desc_lines.append("Count model present but variable '{}' not found in model.".format(var))
        except Exception as e:
            desc_lines.append("Error extracting from count model: {}".format(e))
    else:
        desc_lines.append("No count model (negative binomial / poisson) found in model_output.")

    # Construct a concise conclusion sentence about whether having children decreases engagement in affairs.
    conclusion = "Conclusion: "
    # Prefer logistic model for binary outcome evidence
    if out["logit"] is not None:
        coef = out["logit"].get("coef_children")
        pval = out["logit"].get("pvalue_children")
        if coef is None:
            conclusion += "No usable coefficient for ChildrenYes in logistic model."
        else:
            if (pval is not None) and (pval < alpha):
                if coef < 0:
                    conclusion += "Having children is associated with a statistically significant decrease in the odds of having any extramarital affair (logistic model: coef = {:.4f}, p = {:.3g}, OR = {:.3f}).".format(coef, pval, out["logit"]["odds_ratio_children"])
                elif coef > 0:
                    conclusion += "Having children is associated with a statistically significant increase in the odds of having any extramarital affair (logistic model: coef = {:.4f}, p = {:.3g}, OR = {:.3f}).".format(coef, pval, out["logit"]["odds_ratio_children"])
                else:
                    conclusion += "No association in logistic model (coef = 0)."
            else:
                # not significant
                if coef < 0:
                    conclusion += "Point estimate indicates a decrease in odds (coef = {:.4f}, OR = {:.3f}), but this is not statistically significant (p = {}).".format(coef, out["logit"]["odds_ratio_children"], pval)
                elif coef > 0:
                    conclusion += "Point estimate indicates an increase in odds (coef = {:.4f}, OR = {:.3f}), but this is not statistically significant (p = {}).".format(coef, out["logit"]["odds_ratio_children"], pval)
                else:
                    conclusion += "No association in logistic model (coef = 0)."
    elif out["count"] is not None:
        coef = out["count"].get("coef_children")
        pval = out["count"].get("pvalue_children")
        if coef is None:
            conclusion += "No usable coefficient for ChildrenYes in count model."
        else:
            if (pval is not None) and (pval < alpha):
                if coef < 0:
                    conclusion += "Having children is associated with a statistically significant decrease in the frequency of affairs (count model: coef = {:.4f}, p = {:.3g}, IRR = {:.3f}).".format(coef, pval, out["count"]["irr_children"])
                elif coef > 0:
                    conclusion += "Having children is associated with a statistically significant increase in the frequency of affairs (count model: coef = {:.4f}, p = {:.3g}, IRR = {:.3f}).".format(coef, pval, out["count"]["irr_children"])
                else:
                    conclusion += "No association in count model (coef = 0)."
            else:
                if coef < 0:
                    conclusion += "Point estimate indicates a decrease in frequency (coef = {:.4f}, IRR = {:.3f}), but this is not statistically significant (p = {}).".format(coef, out["count"]["irr_children"], pval)
                elif coef > 0:
                    conclusion += "Point estimate indicates an increase in frequency (coef = {:.4f}, IRR = {:.3f}), but this is not statistically significant (p = {}).".format(coef, out["count"]["irr_children"], pval)
                else:
                    conclusion += "No association in count model (coef = 0)."
    else:
        conclusion += "No relevant models were available to form a conclusion."

    desc_lines.append(conclusion)

    return {"object": out, "description": " ".join(desc_lines)}