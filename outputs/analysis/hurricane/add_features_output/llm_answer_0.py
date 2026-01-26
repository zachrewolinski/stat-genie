def extract_final_answer(model_output):
    """
    Extracts coefficient, uncertainty, and interpretable effect sizes for the 'z_masfem'
    (standardized masculinity-femininity, higher = more feminine) predictor from the
    provided statsmodels result objects.

    Expects model_output to be a dict containing zero or more of these keys:
      - 'ols_log_alldeaths' : OLSResults (robustcov results)
      - 'ols_log_ndam15'    : OLSResults (robustcov results)
      - 'nb_alldeaths'      : GLMResultsWrapper (Negative Binomial) OR
      - 'pois_alldeaths'    : GLMResultsWrapper (Poisson) as a fallback

    Returns a dict with keys:
      - "object": dict of numeric extractions for each model present
      - "description": plain-language interpretation about direction, magnitude,
                       and statistical evidence regarding the hypothesis that
                       more feminine names lead to worse public outcomes (i.e.,
                       positive association with deaths/damages).
    """
    import numpy as np
    import math

    out = {"object": {}, "description": ""}

    def _get_stats(res, var='z_masfem'):
        # extracts coef, se, stat (t or z), pval, conf int, nobs
        stats = {"coef": None, "se": None, "stat": None, "pval": None,
                 "ci_lower": None, "ci_upper": None, "nobs": None}
        if res is None:
            return stats
        params = getattr(res, "params", None)
        bse = getattr(res, "bse", None)
        pvalues = getattr(res, "pvalues", None)
        # try to get statistic (tvalues or zvalues)
        statvals = getattr(res, "tvalues", None)
        if statvals is None:
            statvals = getattr(res, "zvalues", None)
        # nobs
        nobs = getattr(res, "nobs", None)
        try:
            if params is not None and var in params:
                stats["coef"] = float(params[var])
            if bse is not None and var in bse:
                stats["se"] = float(bse[var])
            if statvals is not None and var in statvals:
                stats["stat"] = float(statvals[var])
            if pvalues is not None and var in pvalues:
                stats["pval"] = float(pvalues[var])
            if nobs is not None:
                # ensure integer if possible
                try:
                    stats["nobs"] = int(nobs)
                except Exception:
                    stats["nobs"] = float(nobs)
        except Exception:
            # access by position as fallback
            try:
                idx = list(params.index).index(var)
                stats["coef"] = float(params.iloc[idx])
                stats["se"] = float(bse.iloc[idx]) if bse is not None else None
                stats["stat"] = float(statvals.iloc[idx]) if statvals is not None else None
                stats["pval"] = float(pvalues.iloc[idx]) if pvalues is not None else None
            except Exception:
                pass

        # confidence interval: try res.conf_int()
        try:
            ci = res.conf_int()
            # ci may be DataFrame or ndarray
            if hasattr(ci, "loc") and var in ci.index:
                stats["ci_lower"] = float(ci.loc[var, 0])
                stats["ci_upper"] = float(ci.loc[var, 1])
            else:
                # try position-based (find var index)
                if hasattr(ci, "__len__") and stats["coef"] is not None:
                    # find closest row by matching coef
                    # fallback to normal approximation if cannot find
                    raise Exception
        except Exception:
            # fallback to normal approx using coef +/- 1.96*se (if available)
            if stats["coef"] is not None and stats["se"] is not None:
                stats["ci_lower"] = stats["coef"] - 1.96 * stats["se"]
                stats["ci_upper"] = stats["coef"] + 1.96 * stats["se"]

        return stats

    # Process each expected model key if present
    model_keys = ['ols_log_alldeaths', 'ols_log_ndam15', 'nb_alldeaths', 'pois_alldeaths']
    present_models = [k for k in model_keys if k in model_output]
    if not present_models:
        out["description"] = "No supported model objects found in model_output."
        return out

    summaries = {}
    for key in present_models:
        res = model_output.get(key)
        stats = _get_stats(res, var='z_masfem')
        # Add model-specific interpretable metrics
        info = {"coef": stats["coef"], "se": stats["se"], "stat": stats["stat"],
                "pval": stats["pval"], "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"], "nobs": stats["nobs"]}

        if key.startswith('ols_log_'):
            # coefficient is change in log(outcome) per 1 SD increase in femininity
            if stats["coef"] is not None:
                # multiplicative effect on raw outcome: exp(beta)
                try:
                    mult = math.exp(stats["coef"])
                    pct_change = (mult - 1.0) * 100.0
                except Exception:
                    mult = None
                    pct_change = None
                # CI on multiplicative scale
                if stats["ci_lower"] is not None and stats["ci_upper"] is not None:
                    try:
                        ci_mult_low = math.exp(stats["ci_lower"])
                        ci_mult_high = math.exp(stats["ci_upper"])
                        ci_pct_low = (ci_mult_low - 1.0) * 100.0
                        ci_pct_high = (ci_mult_high - 1.0) * 100.0
                    except Exception:
                        ci_pct_low = ci_pct_high = None
                else:
                    ci_pct_low = ci_pct_high = None
                info.update({
                    "multiplicative_effect": mult,
                    "pct_change": pct_change,
                    "pct_ci_lower": ci_pct_low,
                    "pct_ci_upper": ci_pct_high,
                    "interpretation": (
                        "A 1 SD increase in femininity is associated with a change of "
                        f"{stats['coef']:.4g} in log(outcome), equivalent to a "
                        f"{pct_change:.3g}% change in the outcome (95% CI: "
                        f"{ci_pct_low:.3g}% to {ci_pct_high:.3g}%)"
                        if stats["coef"] is not None else "Insufficient stats"
                    )
                })
        else:
            # count model (negative binomial or poisson): exponentiate to get IRR
            if stats["coef"] is not None:
                try:
                    irr = math.exp(stats["coef"])
                except Exception:
                    irr = None
                if stats["ci_lower"] is not None and stats["ci_upper"] is not None:
                    try:
                        irr_ci_low = math.exp(stats["ci_lower"])
                        irr_ci_high = math.exp(stats["ci_upper"])
                    except Exception:
                        irr_ci_low = irr_ci_high = None
                else:
                    irr_ci_low = irr_ci_high = None
                pct_effect = (irr - 1.0) * 100.0 if irr is not None else None
                info.update({
                    "IRR": irr,
                    "IRR_ci_lower": irr_ci_low,
                    "IRR_ci_upper": irr_ci_high,
                    "pct_change": pct_effect,
                    "interpretation": (
                        "A 1 SD increase in femininity is associated with an IRR = "
                        f"{irr:.4g} ({irr_ci_low:.4g} to {irr_ci_high:.4g}), i.e. "
                        f"{pct_effect:.3g}% change in expected counts"
                        if irr is not None else "Insufficient stats"
                    )
                })

        summaries[key] = info

    out["object"] = summaries

    # Build summary description (evidence for hypothesis)
    # Hypothesis: more feminine names -> fewer precautions -> higher deaths/damages.
    # So we expect positive association between z_masfem and deaths/damages.
    conclusions = []
    positive_significant = 0
    negative_significant = 0
    for k, s in summaries.items():
        coef = s.get("coef")
        p = s.get("pval")
        if coef is None:
            conclusions.append(f"{k}: no estimate extracted.")
            continue
        sig = (p is not None and p < 0.05)
        direction = "positive" if coef > 0 else ("negative" if coef < 0 else "null")
        desc = f"{k}: coef={coef:.4g}, p={p:.3g}" if p is not None else f"{k}: coef={coef:.4g}, p=NA"
        if sig:
            desc += " (statistically significant at p<0.05)"
            if coef > 0:
                positive_significant += 1
            elif coef < 0:
                negative_significant += 1
        else:
            desc += " (not statistically significant)"
        # add effect magnitude interpretation if available
        pct = s.get("pct_change")
        if pct is not None:
            desc += f"; implied % change ~ {pct:.3g}% (95% CI: "
            if "pct_ci_lower" in s and s["pct_ci_lower"] is not None:
                desc += f"{s['pct_ci_lower']:.3g}% to {s['pct_ci_upper']:.3g}%)"
            elif "IRR_ci_lower" in s and s["IRR_ci_lower"] is not None:
                ci_low = (s["IRR_ci_lower"] - 1.0) * 100.0
                ci_high = (s["IRR_ci_upper"] - 1.0) * 100.0
                desc += f"{ci_low:.3g}% to {ci_high:.3g}%)"
            else:
                desc += "CI unavailable)"
        conclusions.append(desc)

    # Decide overall verdict
    if positive_significant > 0 and negative_significant == 0:
        verdict = (
            "Overall: Evidence consistent with the hypothesis. At least one model shows a "
            "statistically significant positive association between name femininity and worse public outcomes "
            "(higher deaths/damages)."
        )
    elif negative_significant > 0 and positive_significant == 0:
        verdict = (
            "Overall: Evidence contradicts the hypothesis. At least one model shows a "
            "statistically significant negative association (more feminine -> lower outcomes)."
        )
    elif positive_significant == 0 and negative_significant == 0:
        verdict = (
            "Overall: No clear evidence. None of the models show a statistically significant association "
            "between name femininity and the outcomes at p<0.05."
        )
    else:
        verdict = (
            "Overall: Mixed evidence. Some models show a significant positive association while others show "
            "a significant negative association."
        )

    out["description"] = verdict + " Details per model: " + " | ".join(conclusions)
    return out