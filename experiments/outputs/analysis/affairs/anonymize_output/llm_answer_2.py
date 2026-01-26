def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'HasChildren' from provided model output.

    Expects model_output to be a dict possibly containing:
      - 'zinb': statsmodels ZeroInflatedNegativeBinomialResultsWrapper
      - 'logit': statsmodels BinaryResultsWrapper (Logit)
      - 'ols_robust': statsmodels RegressionResultsWrapper (OLS with robust SE)

    Returns a dict with:
      - "object": a nested dict with numeric estimates, p-values, CIs, and transformed effects (IRR / odds ratio)
      - "description": a concise interpretation answering whether having children decreases engagement in extramarital affairs,
                       based on sign and statistical significance of coefficients (alpha=0.05).
    """
    import numpy as np
    import math

    out = {"zinb": None, "logit": None, "ols": None}
    desc_lines = []

    def _find_param(res, target='HasChildren', exclude_inflate=False, include_inflate=False):
        # Return parameter name matching target with desired inflation inclusion/exclusion
        names = []
        try:
            idx = list(res.params.index)
        except Exception:
            idx = []
        for n in idx:
            nlow = n.lower()
            if target.lower() in nlow:
                if exclude_inflate and 'inflate' in nlow:
                    continue
                if include_inflate and 'inflate' not in nlow:
                    continue
                names.append(n)
        return names[0] if names else None

    # 1) ZINB
    zinb = model_output.get('zinb')
    if zinb is not None and hasattr(zinb, 'params'):
        try:
            params = zinb.params
            pvals = zinb.pvalues
            conf = zinb.conf_int()
            # count-part parameter for HasChildren (non-inflation)
            name_count = _find_param(zinb, 'HasChildren', exclude_inflate=True)
            zinb_count = None
            if name_count:
                coef = float(params[name_count])
                p = float(pvals[name_count]) if name_count in pvals.index else math.nan
                ci_low, ci_high = map(float, conf.loc[name_count].tolist()) if name_count in conf.index else (math.nan, math.nan)
                irr = float(np.exp(coef))
                irr_ci = (float(np.exp(ci_low)), float(np.exp(ci_high))) if not (math.isnan(ci_low) or math.isnan(ci_high)) else (math.nan, math.nan)
                zinb_count = {
                    "param_name": name_count,
                    "coef": coef,
                    "pvalue": p,
                    "ci_95": (ci_low, ci_high),
                    "incidence_rate_ratio": irr,
                    "irr_95_ci": irr_ci
                }
                out["zinb"] = out["zinb"] or {}
                out["zinb"]["count"] = zinb_count
                # interpret
                if p < 0.05:
                    if coef < 0:
                        desc_lines.append(f"ZINB count: Having children is associated with a statistically significant decrease in affair counts (IRR={irr:.3f}, p={p:.3g}).")
                    else:
                        desc_lines.append(f"ZINB count: Having children is associated with a statistically significant increase in affair counts (IRR={irr:.3f}, p={p:.3g}).")
                else:
                    desc_lines.append(f"ZINB count: No statistically significant association between having children and affair counts (coef={coef:.3f}, p={p:.3g}).")
            # inflation (zero) part parameter for HasChildren (if present)
            name_infl = _find_param(zinb, 'HasChildren', include_inflate=True)
            zinb_infl = None
            if name_infl and 'inflate' in name_infl.lower():
                coef_i = float(params[name_infl])
                p_i = float(pvals[name_infl]) if name_infl in pvals.index else math.nan
                ci_low_i, ci_high_i = map(float, conf.loc[name_infl].tolist()) if name_infl in conf.index else (math.nan, math.nan)
                # inflation is logit for being an excess zero. Positive coef -> higher log-odds of being always-zero (i.e., not at-risk)
                or_infl = float(np.exp(coef_i))
                or_infl_ci = (float(np.exp(ci_low_i)), float(np.exp(ci_high_i))) if not (math.isnan(ci_low_i) or math.isnan(ci_high_i)) else (math.nan, math.nan)
                zinb_infl = {
                    "param_name": name_infl,
                    "coef": coef_i,
                    "pvalue": p_i,
                    "ci_95": (ci_low_i, ci_high_i),
                    "odds_ratio_inflation": or_infl,
                    "or_inflation_95_ci": or_infl_ci
                }
                out["zinb"]["inflation"] = zinb_infl
                # interpret inflation
                if p_i < 0.05:
                    if coef_i > 0:
                        desc_lines.append(f"ZINB inflation: Having children significantly increases the odds of being an 'excess zero' (i.e., not at risk of affairs) (OR={or_infl:.3f}, p={p_i:.3g}).")
                    else:
                        desc_lines.append(f"ZINB inflation: Having children significantly decreases the odds of being an 'excess zero' (OR={or_infl:.3f}, p={p_i:.3g}).")
                else:
                    desc_lines.append(f"ZINB inflation: No significant association in the zero-inflation equation (coef={coef_i:.3f}, p={p_i:.3g}).")
        except Exception as e:
            out["zinb_error"] = str(e)
    else:
        if zinb is not None:
            out["zinb_error"] = "ZINB object present but lacks expected attributes."
        else:
            out["zinb_error"] = "ZINB model not provided."

    # 2) Logit (any-affair)
    logit = model_output.get('logit')
    if logit is not None and hasattr(logit, 'params'):
        try:
            params = logit.params
            pvals = logit.pvalues
            conf = logit.conf_int()
            name = None
            for n in params.index:
                if 'haschildren' in n.lower():
                    name = n
                    break
            if name:
                coef = float(params[name])
                p = float(pvals[name]) if name in pvals.index else math.nan
                ci_low, ci_high = map(float, conf.loc[name].tolist()) if name in conf.index else (math.nan, math.nan)
                odds = float(np.exp(coef))
                odds_ci = (float(np.exp(ci_low)), float(np.exp(ci_high))) if not (math.isnan(ci_low) or math.isnan(ci_high)) else (math.nan, math.nan)
                out["logit"] = {
                    "param_name": name,
                    "coef": coef,
                    "pvalue": p,
                    "ci_95": (ci_low, ci_high),
                    "odds_ratio": odds,
                    "odds_ratio_95_ci": odds_ci
                }
                if p < 0.05:
                    if coef < 0:
                        desc_lines.append(f"Logit: Having children is associated with significantly lower odds of any affair (OR={odds:.3f}, p={p:.3g}).")
                    else:
                        desc_lines.append(f"Logit: Having children is associated with significantly higher odds of any affair (OR={odds:.3f}, p={p:.3g}).")
                else:
                    desc_lines.append(f"Logit: No statistically significant association between having children and the probability of any affair (coef={coef:.3f}, p={p:.3g}).")
            else:
                out["logit_error"] = "No parameter matching 'HasChildren' found in logit model."
        except Exception as e:
            out["logit_error"] = str(e)
    else:
        if logit is not None:
            out["logit_error"] = "Logit object present but lacks expected attributes."
        else:
            out["logit_error"] = "Logit model not provided."

    # 3) OLS (robust)
    ols = model_output.get('ols_robust')
    if ols is not None and hasattr(ols, 'params'):
        try:
            params = ols.params
            pvals = ols.pvalues
            conf = ols.conf_int()
            name = None
            for n in params.index:
                if 'haschildren' in n.lower():
                    name = n
                    break
            if name:
                coef = float(params[name])
                p = float(pvals[name]) if name in pvals.index else math.nan
                ci_low, ci_high = map(float, conf.loc[name].tolist()) if name in conf.index else (math.nan, math.nan)
                out["ols"] = {
                    "param_name": name,
                    "coef": coef,
                    "pvalue": p,
                    "ci_95": (ci_low, ci_high)
                }
                if p < 0.05:
                    if coef < 0:
                        desc_lines.append(f"OLS: Having children is associated with a statistically significant decrease in affair counts (coef={coef:.3f}, p={p:.3g}).")
                    else:
                        desc_lines.append(f"OLS: Having children is associated with a statistically significant increase in affair counts (coef={coef:.3f}, p={p:.3g}).")
                else:
                    desc_lines.append(f"OLS: No statistically significant association detected (coef={coef:.3f}, p={p:.3g}).")
            else:
                out["ols_error"] = "No parameter matching 'HasChildren' found in OLS model."
        except Exception as e:
            out["ols_error"] = str(e)
    else:
        if ols is not None:
            out["ols_error"] = "OLS object present but lacks expected attributes."
        else:
            out["ols_error"] = "OLS model not provided."

    # Final concise conclusion based on available significant results:
    # Prefer ZINB count and logit for substantive conclusions; fall back to OLS.
    conclusion = "Based on models: "
    sig_negative = 0
    sig_positive = 0
    evidence_msgs = []

    # Check ZINB count
    zn = out.get("zinb") or {}
    if isinstance(zn, dict) and zn.get("count"):
        c = zn["count"]
        if not math.isnan(c["pvalue"]) and c["pvalue"] < 0.05:
            if c["coef"] < 0:
                sig_negative += 1
                evidence_msgs.append(f"ZINB count IRR={c['incidence_rate_ratio']:.3f} (p={c['pvalue']:.3g})")
            else:
                sig_positive += 1
                evidence_msgs.append(f"ZINB count IRR={c['incidence_rate_ratio']:.3f} (p={c['pvalue']:.3g})")
    # Check logit
    lg = out.get("logit")
    if isinstance(lg, dict):
        if not math.isnan(lg["pvalue"]) and lg["pvalue"] < 0.05:
            if lg["coef"] < 0:
                sig_negative += 1
                evidence_msgs.append(f"Logit OR={lg['odds_ratio']:.3f} (p={lg['pvalue']:.3g})")
            else:
                sig_positive += 1
                evidence_msgs.append(f"Logit OR={lg['odds_ratio']:.3f} (p={lg['pvalue']:.3g})")
    # Check OLS
    ol = out.get("ols")
    if isinstance(ol, dict):
        if not math.isnan(ol["pvalue"]) and ol["pvalue"] < 0.05:
            if ol["coef"] < 0:
                sig_negative += 1
                evidence_msgs.append(f"OLS coef={ol['coef']:.3f} (p={ol['pvalue']:.3g})")
            else:
                sig_positive += 1
                evidence_msgs.append(f"OLS coef={ol['coef']:.3f} (p={ol['pvalue']:.3g})")

    if sig_negative > sig_positive and sig_negative > 0:
        conclusion += "Overall, having children is associated with a decrease in engagement in extramarital affairs (supported by multiple models: " + "; ".join(evidence_msgs) + ")."
    elif sig_positive > sig_negative and sig_positive > 0:
        conclusion += "Overall, having children is associated with an increase in engagement in extramarital affairs (supported by multiple models: " + "; ".join(evidence_msgs) + ")."
    else:
        # No consistent significant effect
        if evidence_msgs:
            conclusion += "Mixed evidence; some models show significant effects but not consistently in one direction: " + "; ".join(evidence_msgs) + "."
        else:
            conclusion += "No robust evidence that having children changes engagement in extramarital affairs (no consistent statistically significant effects at alpha=0.05)."

    # Build final description: short summary + model-specific notes (first few lines)
    description = "\n".join(desc_lines[:6])  # keep concise
    if description:
        description = description + "\n\n" + conclusion
    else:
        description = conclusion

    return {"object": out, "description": description}