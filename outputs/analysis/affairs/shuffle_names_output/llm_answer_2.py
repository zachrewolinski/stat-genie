def extract_final_answer(model_output):
    """
    Extract statistics about the effect of 'HasChildren' on 'NumAffairs' from the
    model_output produced by the provided modeling function.

    Returns a dict with:
      - "object": a dict containing extracted numeric results (coefficients, SEs,
                  p-values, CIs, IRR for negbin, group means)
      - "description": a short plain-language interpretation of those numbers
                       in the context of whether having children decreases
                       engagement in extramarital affairs.
    """
    import numpy as np

    out = {
        "negbin": None,
        "ols": None,
        "group_summary": None,
        "errors": {}
    }

    # Helper to safely pull parameter stats for a given results object
    def _pull_params(res, varname):
        try:
            params = res.params
            pvals = res.pvalues
            bse = res.bse
            ci = res.conf_int()
        except Exception as e:
            raise RuntimeError(f"Could not extract stats from results object: {e}")

        if varname not in params.index:
            raise KeyError(f"Variable '{varname}' not found in model parameters. "
                           f"Available params: {list(params.index)}")

        coef = float(params[varname])
        se = float(bse[varname]) if varname in bse.index else None
        pval = float(pvals[varname]) if varname in pvals.index else None
        ci_low = float(ci.loc[varname, 0]) if varname in ci.index else None
        ci_high = float(ci.loc[varname, 1]) if varname in ci.index else None

        return {"coef": coef, "se": se, "pvalue": pval, "ci_low": ci_low, "ci_high": ci_high}

    # 1) Negative binomial results
    if "negbin" in model_output and model_output.get("negbin") is not None:
        try:
            nb_res = model_output["negbin"]
            nb_stats = _pull_params(nb_res, "HasChildren")
            # For count models, exponentiate to get incidence rate ratio (IRR)
            irr = float(np.exp(nb_stats["coef"]))
            irr_ci_low = float(np.exp(nb_stats["ci_low"])) if nb_stats["ci_low"] is not None else None
            irr_ci_high = float(np.exp(nb_stats["ci_high"])) if nb_stats["ci_high"] is not None else None
            pct_change = (irr - 1.0) * 100.0  # percent change in expected count
            out["negbin"] = {
                "coef_log": nb_stats["coef"],
                "se": nb_stats["se"],
                "pvalue": nb_stats["pvalue"],
                "ci_log": [nb_stats["ci_low"], nb_stats["ci_high"]],
                "irr": irr,
                "irr_ci": [irr_ci_low, irr_ci_high],
                "irr_pct_change": pct_change
            }
        except Exception as e:
            out["errors"]["negbin"] = str(e)
    elif "negbin_error" in model_output:
        out["errors"]["negbin"] = model_output.get("negbin_error")

    # 2) OLS results
    if "ols" in model_output and model_output.get("ols") is not None:
        try:
            ols_res = model_output["ols"]
            ols_stats = _pull_params(ols_res, "HasChildren")
            ci_low = ols_stats["ci_low"]
            ci_high = ols_stats["ci_high"]
            out["ols"] = {
                "coef": ols_stats["coef"],
                "se": ols_stats["se"],
                "pvalue": ols_stats["pvalue"],
                "ci": [ci_low, ci_high],
                "interpretation": ("An OLS coefficient is the expected absolute change in "
                                   "the NumAffairs score associated with HasChildren=1 vs 0, "
                                   "holding controls constant.")
            }
        except Exception as e:
            out["errors"]["ols"] = str(e)
    elif "ols_error" in model_output:
        out["errors"]["ols"] = model_output.get("ols_error")

    # 3) Group summary (unadjusted means)
    if "group_summary_NumAffairs_by_HasChildren" in model_output:
        try:
            gs = model_output["group_summary_NumAffairs_by_HasChildren"]
            # expected MultiIndex columns like ('NumAffairs','mean'), etc.
            # We'll try to pick count, mean, std, median for groups 0 and 1 if present.
            group_info = {}
            for hasch in gs.index:
                try:
                    count = int(gs.loc[hasch, ("NumAffairs", "count")])
                    mean = float(gs.loc[hasch, ("NumAffairs", "mean")])
                    std = float(gs.loc[hasch, ("NumAffairs", "std")]) if ("std") in gs.columns.get_level_values(1) else None
                    median = float(gs.loc[hasch, ("NumAffairs", "median")]) if ("median") in gs.columns.get_level_values(1) else None
                except Exception:
                    # fallback if columns are single-level
                    try:
                        count = int(gs.loc[hasch, "count"])
                        mean = float(gs.loc[hasch, "mean"])
                        std = float(gs.loc[hasch, "std"]) if "std" in gs.columns else None
                        median = float(gs.loc[hasch, "median"]) if "median" in gs.columns else None
                    except Exception:
                        raise
                group_info[int(hasch)] = {"n": count, "mean_NumAffairs": mean, "std": std, "median": median}
            out["group_summary"] = group_info
        except Exception as e:
            out["errors"]["group_summary"] = str(e)

    # 4) Short, actionable conclusion based on present stats
    concl_parts = []
    # Prefer negbin for inference, fallback to OLS if negbin missing
    if out["negbin"]:
        try:
            irr = out["negbin"]["irr"]
            p = out["negbin"]["pvalue"]
            pct = out["negbin"]["irr_pct_change"]
            sig = (p is not None and p < 0.05)
            direction = "decrease" if irr < 1 else "increase" if irr > 1 else "no change"
            concl_parts.append(
                f"Negative-binomial: HasChildren is associated with a {pct:.1f}% "
                f"{direction} in the expected number of reported extramarital acts "
                f"(IRR={irr:.3f}, 95% CI={out['negbin']['irr_ci']}, p={p:.3g}). "
                f"{'Statistically significant.' if sig else 'Not statistically significant.'}"
            )
        except Exception:
            pass

    if out["ols"]:
        try:
            coef = out["ols"]["coef"]
            p = out["ols"]["pvalue"]
            sig = (p is not None and p < 0.05)
            direction = "fewer" if coef < 0 else "more" if coef > 0 else "no change"
            concl_parts.append(
                f"OLS: HasChildren coefficient = {coef:.3f} (95% CI={out['ols']['ci']}, p={p:.3g}); "
                f"this suggests {direction} reported affairs associated with children. "
                f"{'Statistically significant.' if sig else 'Not statistically significant.'}"
            )
        except Exception:
            pass

    if out["group_summary"]:
        try:
            g0 = out["group_summary"].get(0)
            g1 = out["group_summary"].get(1)
            if g0 is not None and g1 is not None:
                concl_parts.append(
                    f"Unadjusted means: mean NumAffairs without children = {g0['mean_NumAffairs']:.3f} "
                    f"(n={g0['n']}); with children = {g1['mean_NumAffairs']:.3f} (n={g1['n']})."
                )
        except Exception:
            pass

    if not concl_parts:
        concl = "No usable model statistics were found in model_output to form a conclusion."
    else:
        concl = " ".join(concl_parts)

    return {
        "object": out,
        "description": (
            "Extracted statistics for the effect of HasChildren on NumAffairs. "
            "The 'object' field contains: negbin results (log-coef, SE, p-value, 95% CI, IRR and IRR CI), "
            "OLS results (coef, SE, p-value, 95% CI), unadjusted group means by HasChildren, and any extraction errors. "
            "The 'conclusion' is a brief plain-language summary appended inside the 'object' under the 'errors'/summary info "
            "and also returned here as text. Interpret IRR < 1 as a lower expected count of affairs when HasChildren=1; "
            "for OLS the coefficient gives the absolute change in NumAffairs associated with HasChildren=1 vs 0."
        ) + " Conclusion summary: " + concl
    }