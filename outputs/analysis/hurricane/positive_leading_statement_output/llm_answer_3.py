def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs and interpretable effect sizes
    for the predictor 'masfem_z' from both the negative-binomial (fatalities) and OLS
    (log damage) results in model_output.

    Returns:
      {
        "object": {
          "deaths": {coef, se, pval, ci_lower, ci_upper, irr, irr_ci_lower, irr_ci_upper, "significant": bool},
          "damage": {coef, se, pval, ci_lower, ci_upper, pct_change, pct_change_ci_lower, pct_change_ci_upper, "significant": bool}
        },
        "description": "<text interpretation>"
      }
    """
    import numpy as np

    out = {"deaths": None, "damage": None}

    def _get_stats(res, varname):
        # Extract coef, se, pval, conf int robustly
        if not hasattr(res, "params"):
            raise ValueError("Result object has no params attribute")
        params = res.params
        if varname not in params.index:
            raise KeyError(f"{varname} not found in result.params index")
        coef = float(params[varname])
        se = float(res.bse[varname]) if hasattr(res, "bse") else None
        pval = float(res.pvalues[varname]) if hasattr(res, "pvalues") else None

        # confidence interval extraction (works if conf_int returns DataFrame/ndarray)
        conf = res.conf_int()
        try:
            # try label-based access
            ci_low = float(conf.loc[varname][0])
            ci_high = float(conf.loc[varname][1])
        except Exception:
            # fallback to positional access
            try:
                idx = list(params.index).index(varname)
                ci_low = float(conf[idx, 0])
                ci_high = float(conf[idx, 1])
            except Exception:
                ci_low = None
                ci_high = None

        return {"coef": coef, "se": se, "pval": pval, "ci_lower": ci_low, "ci_upper": ci_high}

    # 1) deaths (negative binomial)
    if model_output.get("deaths_nb") is not None:
        nb_res = model_output["deaths_nb"]
        try:
            stats_deaths = _get_stats(nb_res, "masfem_z")
            # incidence rate ratio (IRR) and CI
            irr = float(np.exp(stats_deaths["coef"]))
            irr_ci_lower = float(np.exp(stats_deaths["ci_lower"])) if stats_deaths["ci_lower"] is not None else None
            irr_ci_upper = float(np.exp(stats_deaths["ci_upper"])) if stats_deaths["ci_upper"] is not None else None
            significant = (stats_deaths["pval"] is not None) and (stats_deaths["pval"] < 0.05)

            out["deaths"] = {
                "coef": stats_deaths["coef"],
                "se": stats_deaths["se"],
                "pval": stats_deaths["pval"],
                "ci_lower": stats_deaths["ci_lower"],
                "ci_upper": stats_deaths["ci_upper"],
                "irr": irr,
                "irr_ci_lower": irr_ci_lower,
                "irr_ci_upper": irr_ci_upper,
                "significant": bool(significant),
            }
        except Exception as e:
            out["deaths"] = {"error": str(e)}
    else:
        out["deaths"] = None

    # 2) damage (OLS on log damage)
    if model_output.get("damage_ols") is not None:
        ols_res = model_output["damage_ols"]
        try:
            stats_damage = _get_stats(ols_res, "masfem_z")
            # Convert log-coefficient to percent change: (exp(beta)-1)*100
            pct_change = (np.exp(stats_damage["coef"]) - 1.0) * 100.0
            pct_ci_lower = (np.exp(stats_damage["ci_lower"]) - 1.0) * 100.0 if stats_damage["ci_lower"] is not None else None
            pct_ci_upper = (np.exp(stats_damage["ci_upper"]) - 1.0) * 100.0 if stats_damage["ci_upper"] is not None else None
            significant = (stats_damage["pval"] is not None) and (stats_damage["pval"] < 0.05)

            out["damage"] = {
                "coef_log": stats_damage["coef"],
                "se": stats_damage["se"],
                "pval": stats_damage["pval"],
                "ci_lower_log": stats_damage["ci_lower"],
                "ci_upper_log": stats_damage["ci_upper"],
                "pct_change": pct_change,
                "pct_change_ci_lower": pct_ci_lower,
                "pct_change_ci_upper": pct_ci_upper,
                "significant": bool(significant),
            }
        except Exception as e:
            out["damage"] = {"error": str(e)}
    else:
        out["damage"] = None

    # Build a concise interpretation
    desc_parts = []
    if out["deaths"] and "error" not in out["deaths"]:
        d = out["deaths"]
        desc_parts.append(
            "Fatalities (Negative Binomial): masfem_z coef = {coef:.4f} (SE={se:.4f}), p={pval:.3g}, "
            "95%CI=[{lo:.4f}, {hi:.4f}]. IRR = {irr:.3f} (95%CI [{irrl:.3f}, {irrh:.3f}]). {sig}".format(
                coef=d["coef"], se=d["se"] if d["se"] is not None else float("nan"),
                pval=d["pval"] if d["pval"] is not None else float("nan"),
                lo=d["ci_lower"] if d["ci_lower"] is not None else float("nan"),
                hi=d["ci_upper"] if d["ci_upper"] is not None else float("nan"),
                irr=d["irr"] if d["irr"] is not None else float("nan"),
                irrl=d["irr_ci_lower"] if d["irr_ci_lower"] is not None else float("nan"),
                irrh=d["irr_ci_upper"] if d["irr_ci_upper"] is not None else float("nan"),
                sig=("Statistically significant (p<0.05)." if d["significant"] else "Not statistically significant.")
            )
        )
    else:
        desc_parts.append("Fatalities result unavailable or error: {}".format(out["deaths"].get("error") if out["deaths"] else "None"))

    if out["damage"] and "error" not in out["damage"]:
        dd = out["damage"]
        desc_parts.append(
            "Property damage (OLS on log): masfem_z coef (log) = {coef:.4f} (SE={se:.4f}), p={pval:.3g}, "
            "95%CI_log=[{lo:.4f}, {hi:.4f}]. Interpreted as {pct:.2f}% change in damage per 1-SD increase in name femininity "
            "(95%CI [{pctlo:.2f}%, {pcthi:.2f}%]). {sig}".format(
                coef=dd["coef_log"], se=dd["se"] if dd["se"] is not None else float("nan"),
                pval=dd["pval"] if dd["pval"] is not None else float("nan"),
                lo=dd["ci_lower_log"] if dd["ci_lower_log"] is not None else float("nan"),
                hi=dd["ci_upper_log"] if dd["ci_upper_log"] is not None else float("nan"),
                pct=dd["pct_change"] if dd["pct_change"] is not None else float("nan"),
                pctlo=dd["pct_change_ci_lower"] if dd["pct_change_ci_lower"] is not None else float("nan"),
                pcthi=dd["pct_change_ci_upper"] if dd["pct_change_ci_upper"] is not None else float("nan"),
                sig=("Statistically significant (p<0.05)." if dd["significant"] else "Not statistically significant.")
            )
        )
    else:
        desc_parts.append("Damage result unavailable or error: {}".format(out["damage"].get("error") if out["damage"] else "None"))

    # Overall short conclusion about the hypothesis:
    # Hypothesis: more feminine names -> less precaution -> worse outcomes (more deaths, more damage)
    concl = []
    if out["deaths"] and "error" not in out["deaths"]:
        if out["deaths"]["significant"] and out["deaths"]["coef"] > 0:
            concl.append("Fatalities: result is consistent with the hypothesis (higher femininity -> more deaths; statistically significant).")
        elif out["deaths"]["significant"] and out["deaths"]["coef"] < 0:
            concl.append("Fatalities: statistically significant but in the opposite direction (higher femininity -> fewer deaths).")
        else:
            concl.append("Fatalities: no statistically significant evidence relating femininity to deaths.")
    if out["damage"] and "error" not in out["damage"]:
        if out["damage"]["significant"] and out["damage"]["coef_log"] > 0:
            concl.append("Damage: result is consistent with the hypothesis (higher femininity -> higher logged damage; statistically significant).")
        elif out["damage"]["significant"] and out["damage"]["coef_log"] < 0:
            concl.append("Damage: statistically significant but in the opposite direction (higher femininity -> lower damage).")
        else:
            concl.append("Damage: no statistically significant evidence relating femininity to logged damage.")

    description = " ".join(desc_parts) + " Overall: " + " ".join(concl)

    return {"object": out, "description": description}