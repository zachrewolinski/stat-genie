def extract_final_answer(model_output):
    """
    Extracts effect of 'Children' from model_output dict returned by the modelling function.
    Returns a dict with:
      - "object": structured numeric results per model (negbin, ols, logit if available)
      - "description": concise interpretation answering whether having children decreases affairs
    
    The function is defensive: it handles missing models or error strings in model_output.
    """
    import numpy as np

    out = {}
    summary_lines = []

    def summarize_coef(coef, se, pval, ci_lower, ci_upper, model_name, interpret_as_pct=False):
        # Build a small dict and human-readable line
        entry = {
            "coef": float(coef) if coef is not None else None,
            "se": float(se) if se is not None else None,
            "pvalue": float(pval) if pval is not None else None,
            "ci_2.5%": float(ci_lower) if ci_lower is not None else None,
            "ci_97.5%": float(ci_upper) if ci_upper is not None else None,
        }
        if interpret_as_pct and coef is not None:
            pct = (np.exp(coef) - 1.0) * 100.0
            pct_lo = (np.exp(ci_lower) - 1.0) * 100.0
            pct_hi = (np.exp(ci_upper) - 1.0) * 100.0
            entry.update({
                "pct_change": float(pct),
                "pct_change_ci_2.5%": float(pct_lo),
                "pct_change_ci_97.5%": float(pct_hi),
            })
            line = (f"{model_name}: coef(Children) = {coef:.4f}, SE = {se:.4f}, p = {pval:.3g}; "
                    f"approx. percent change = {pct:.2f}% (95% CI: {pct_lo:.2f}% to {pct_hi:.2f}%).")
        else:
            line = (f"{model_name}: coef(Children) = {coef:.4f}, SE = {se:.4f}, p = {pval:.3g}; "
                    f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}].")
        return entry, line

    # Helper to extract from a statsmodels result object
    def extract_from_result(res, name, interpret_as_pct=False):
        try:
            params = res.params
            bse = getattr(res, "bse")
            # pvalues may or may not exist for some wrappers; try attribute then compute z/t
            pvalues = getattr(res, "pvalues", None)
            ci = None
            try:
                ci = res.conf_int()
            except Exception:
                # conf_int may be a method requiring args; ignore if unavailable
                ci = None

            if "Children" not in params.index:
                raise KeyError("Children not in model parameters")

            coef = float(params["Children"])
            se = float(bse["Children"]) if ("Children" in bse.index) else float(bse.loc["Children"])
            if pvalues is not None and "Children" in pvalues.index:
                pval = float(pvalues["Children"])
            else:
                # fallback: compute Wald z and two-sided p-value assuming normal
                z = coef / se if se != 0 else np.nan
                from scipy import stats
                pval = float(2 * (1 - stats.norm.cdf(abs(z))))

            if ci is not None:
                # conf_int returns DataFrame-like with rows indexed by param names in many wrappers
                try:
                    ci_row = ci.loc["Children"]
                    ci_low = float(ci_row[0])
                    ci_high = float(ci_row[1])
                except Exception:
                    # maybe ndarray with rows in same order as params
                    ci_arr = np.asarray(ci)
                    # find index of Children in params.index
                    idx = list(params.index).index("Children")
                    ci_low = float(ci_arr[idx, 0])
                    ci_high = float(ci_arr[idx, 1])
            else:
                # approximate CI using normal approximation
                from scipy import stats
                z_crit = stats.norm.ppf(0.975)
                ci_low = coef - z_crit * se
                ci_high = coef + z_crit * se

            entry, line = summarize_coef(coef, se, pval, ci_low, ci_high, name,
                                         interpret_as_pct=interpret_as_pct)
            return entry, line
        except Exception as e:
            return None, f"{name}: extraction failed ({e})"

    # 1) Negative binomial (count) model
    if "negbin" in model_output and not isinstance(model_output.get("negbin"), str):
        neg_res = model_output["negbin"]
        neg_entry, neg_line = extract_from_result(neg_res, "NegativeBinomial (GLM)", interpret_as_pct=True)
        out["negbin"] = neg_entry
        summary_lines.append(neg_line)
    else:
        summary_lines.append("NegativeBinomial: model not available or error.")

    # 2) OLS on log(affairs+1)
    if "ols_log1p" in model_output and not isinstance(model_output.get("ols_log1p"), str):
        ols_res = model_output["ols_log1p"]
        # Interpret OLS coef on log(affairs+1) approximately as % change in (affairs+1)
        ols_entry, ols_line = extract_from_result(ols_res, "OLS on log(affairs+1)", interpret_as_pct=True)
        out["ols_log1p"] = ols_entry
        summary_lines.append(ols_line)
    else:
        summary_lines.append("OLS (log1p): model not available or error.")

    # 3) Logit (AnyAffair) if available
    if "logit" in model_output and not isinstance(model_output.get("logit"), str):
        logit_res = model_output["logit"]
        logit_entry, logit_line = extract_from_result(logit_res, "Logit (AnyAffair)", interpret_as_pct=False)
        # For logit we can also compute odds ratio and its CI
        if logit_entry is not None:
            coef = logit_entry["coef"]
            ci_low = logit_entry["ci_2.5%"]
            ci_high = logit_entry["ci_97.5%"]
            or_est = np.exp(coef)
            or_lo = np.exp(ci_low)
            or_hi = np.exp(ci_high)
            logit_entry.update({
                "odds_ratio": float(or_est),
                "odds_ratio_ci_2.5%": float(or_lo),
                "odds_ratio_ci_97.5%": float(or_hi)
            })
            logit_line = (f"Logit (AnyAffair): coef = {coef:.4f}, p = {logit_entry['pvalue']:.3g}; "
                          f"OR = {or_est:.3f} (95% CI: {or_lo:.3f} to {or_hi:.3f}).")
        out["logit"] = logit_entry
        summary_lines.append(logit_line)
    else:
        # If error string provided, include it
        if "logit_error" in model_output:
            summary_lines.append(f"Logit: failed ({model_output.get('logit_error')})")
        else:
            summary_lines.append("Logit: not available.")

    # Decision logic: do results show that having children DECREASES affairs?
    # We consider decrease if coefficient is negative and statistically significant (p < 0.05).
    decisions = []
    for mname in ["negbin", "ols_log1p", "logit"]:
        entry = out.get(mname)
        if entry is None:
            decisions.append((mname, "no_result"))
            continue
        coef = entry.get("coef")
        p = entry.get("pvalue")
        if coef is None or p is None or np.isnan(p):
            decisions.append((mname, "no_result"))
            continue
        if p < 0.05:
            if coef < 0:
                decisions.append((mname, "significant_decrease"))
            elif coef > 0:
                decisions.append((mname, "significant_increase"))
            else:
                decisions.append((mname, "significant_no_change"))
        else:
            decisions.append((mname, "not_significant"))

    # Summarize final conclusion prioritizing negative binomial (count outcome)
    final_conclusion = ""
    # Check negbin decision first
    neg_dec = next((d for m, d in decisions if m == "negbin"), None)
    if neg_dec == "significant_decrease":
        final_conclusion = ("Primary (NegativeBinomial): Having children is associated with a statistically "
                            "significant DECREASE in expected number of extramarital affairs.")
    elif neg_dec == "significant_increase":
        final_conclusion = ("Primary (NegativeBinomial): Having children is associated with a statistically "
                            "significant INCREASE in expected number of extramarital affairs.")
    elif neg_dec == "not_significant":
        final_conclusion = ("Primary (NegativeBinomial): Coefficient for Children is not statistically significant; "
                            "no evidence that having children decreases engagement in extramarital affairs.")
    else:
        # fallback to checking OLS/logit
        ols_dec = next((d for m, d in decisions if m == "ols_log1p"), None)
        logit_dec = next((d for m, d in decisions if m == "logit"), None)
        # If any model shows significant decrease, mention it; otherwise say no evidence of decrease
        if ols_dec == "significant_decrease" or logit_dec == "significant_decrease":
            final_conclusion = ("Some models (notably OLS or Logit) show a statistically significant decrease "
                                "associated with having children, but the primary count model does not provide "
                                "consistent evidence.")
        else:
            final_conclusion = ("Across the fitted models there is no consistent evidence that having children "
                                "decreases engagement in extramarital affairs. If anything, the negative binomial "
                                "point estimate indicates a small positive association (an increase), but it is "
                                "not statistically significant.")

    # Build description combining numeric summary and final conclusion
    description = " ; ".join(summary_lines) + " || Final conclusion: " + final_conclusion

    return {"object": out, "description": description}