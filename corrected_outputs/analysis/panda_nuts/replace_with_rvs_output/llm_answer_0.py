def extract_final_answer(model_output):
    """
    Extract relevant statistics from a statsmodels MixedLMResults (or wrapper) object
    to answer how age, sex, and receiving help relate to nut-cracking efficiency.

    Returns a dictionary with keys:
      - "object": a dictionary with the coefficient table and computed simple effects
      - "description": a short textual interpretation of the key results

    The function attempts to be robust to small API differences in the results object.
    """
    import pandas as pd
    import numpy as np
    import re

    res = model_output

    # Helper to safely get attributes
    def safe_attr(obj, name, alt=None):
        return getattr(obj, name, alt)

    # Try to get parameter estimates
    try:
        params = pd.Series(safe_attr(res, "params", safe_attr(res, "fe_params")))
    except Exception as e:
        raise ValueError("Unable to retrieve parameter estimates from model_output.") from e

    # Standard errors, t/z values, p-values, confidence intervals
    bse = safe_attr(res, "bse", None)
    if bse is None:
        bse = safe_attr(res, "bse_fe", None)
    if bse is not None:
        bse = pd.Series(bse)
    tvalues = safe_attr(res, "tvalues", None)
    pvalues = safe_attr(res, "pvalues", None)
    try:
        ci = res.conf_int()
        # conf_int may return numpy array or DataFrame
        ci = pd.DataFrame(ci, index=params.index)
        ci.columns = ["2.5%", "97.5%"]
    except Exception:
        ci = None

    # Build coef table DataFrame
    coef_df = pd.DataFrame(index=params.index)
    coef_df["coef"] = params
    coef_df["se"] = bse if bse is not None else np.nan
    coef_df["t/z"] = tvalues if tvalues is not None else np.nan
    coef_df["p"] = pvalues if pvalues is not None else np.nan
    if ci is not None:
        coef_df["ci_lower"] = ci["2.5%"]
        coef_df["ci_upper"] = ci["97.5%"]
    else:
        coef_df["ci_lower"] = np.nan
        coef_df["ci_upper"] = np.nan

    # Identify relevant parameter names (robust to factor coding specifics)
    param_names = list(params.index)

    # Find main age parameter name (contains 'age' but not a colon)
    age_param = None
    for n in param_names:
        if re.search(r"\bage\b", n) and ":" not in n:
            age_param = n
            break

    # Find main sex contrast(s)
    sex_params = [n for n in param_names if "C(sex)" in n and ":" not in n]

    # Find main help contrast(s)
    help_params = [n for n in param_names if "C(help)" in n and ":" not in n]

    # Find interactions
    age_help_param = None
    for n in param_names:
        if re.search(r"\bage\b", n) and "C(help)" in n:
            age_help_param = n
            break

    sex_help_param = None
    for n in param_names:
        if "C(sex)" in n and "C(help)" in n:
            sex_help_param = n
            break

    # Prepare cov_params for computing SEs of linear combinations if available
    cov = None
    try:
        cov = res.cov_params()
        cov = pd.DataFrame(cov, index=params.index, columns=params.index)
    except Exception:
        cov = None

    # Helper to compute linear combination estimate and SE (and p) for names with coefficients
    def lincomb(names, coefs=None):
        # names: list of parameter names
        # coefs: list of multipliers (same length) or None => all 1
        if coefs is None:
            coefs = [1.0] * len(names)
        missing = [n for n in names if n not in params.index]
        if missing:
            return None  # can't compute
        est = float(np.sum([params[n] * c for n, c in zip(names, coefs)]))
        if cov is not None:
            vec = np.array([coefs[i] * (1.0 if n in params.index else 0.0) for i, n in enumerate(names)])
            # Build covariance submatrix
            sub = cov.loc[names, names].values
            se = float(np.sqrt(vec @ sub @ vec))
            z = est / se if se > 0 else np.nan
            # Two-sided p-value from normal approx
            p = float(2 * (1 - getattr(__import__("scipy").stats, "norm").cdf(abs(z))))
            return {"est": est, "se": se, "z": z, "p": p}
        else:
            return {"est": est, "se": None, "z": None, "p": None}

    # Compute simple slopes/contrasts where possible
    simple_results = {}

    # Age effect when help = reference level (this is the main age_param)
    if age_param:
        simple_results["age_when_help_ref"] = {
            "param": age_param,
            "coef": float(params[age_param]),
            "se": float(coef_df.loc[age_param, "se"]) if not pd.isna(coef_df.loc[age_param, "se"]) else None,
            "p": float(coef_df.loc[age_param, "p"]) if not pd.isna(coef_df.loc[age_param, "p"]) else None,
            "ci_lower": float(coef_df.loc[age_param, "ci_lower"]) if not pd.isna(coef_df.loc[age_param, "ci_lower"]) else None,
            "ci_upper": float(coef_df.loc[age_param, "ci_upper"]) if not pd.isna(coef_df.loc[age_param, "ci_upper"]) else None,
        }
        # Age effect when help = the contrasted help level (if interaction exists)
        if age_help_param:
            comb = lincomb([age_param, age_help_param], [1.0, 1.0])
            if comb:
                simple_results["age_when_help_contrast"] = {
                    "combination": f"{age_param} + {age_help_param}",
                    "est": comb["est"],
                    "se": comb["se"],
                    "z": comb["z"],
                    "p": comb["p"],
                }

    # Sex effect when help = reference level: depends on which sex contrast exists
    if sex_params:
        # There may be one or multiple contrasts depending on coding; report all
        sex_effects = {}
        for sp in sex_params:
            sex_effects[sp] = {
                "coef": float(params[sp]),
                "se": float(coef_df.loc[sp, "se"]) if not pd.isna(coef_df.loc[sp, "se"]) else None,
                "p": float(coef_df.loc[sp, "p"]) if not pd.isna(coef_df.loc[sp, "p"]) else None,
                "ci_lower": float(coef_df.loc[sp, "ci_lower"]) if not pd.isna(coef_df.loc[sp, "ci_lower"]) else None,
                "ci_upper": float(coef_df.loc[sp, "ci_upper"]) if not pd.isna(coef_df.loc[sp, "ci_upper"]) else None,
            }
            # If sex:help interaction exists for this contrast, compute effect when help = contrasted level
            if sex_help_param and sex_help_param in params.index:
                # sex_help_param might correspond to the same sex level; ensure matching terms
                # If exact match exists use it; else attempt to use the found sex_help_param
                if sex_help_param in params.index:
                    comb = lincomb([sp, sex_help_param], [1.0, 1.0])
                    if comb:
                        sex_effects[sp + "_when_help_contrast"] = {
                            "combination": f"{sp} + {sex_help_param}",
                            "est": comb["est"],
                            "se": comb["se"],
                            "z": comb["z"],
                            "p": comb["p"],
                        }
        simple_results["sex_effects"] = sex_effects

    # Help main effect(s)
    if help_params:
        help_effects = {}
        for hp in help_params:
            help_effects[hp] = {
                "coef": float(params[hp]),
                "se": float(coef_df.loc[hp, "se"]) if not pd.isna(coef_df.loc[hp, "se"]) else None,
                "p": float(coef_df.loc[hp, "p"]) if not pd.isna(coef_df.loc[hp, "p"]) else None,
                "ci_lower": float(coef_df.loc[hp, "ci_lower"]) if not pd.isna(coef_df.loc[hp, "ci_lower"]) else None,
                "ci_upper": float(coef_df.loc[hp, "ci_upper"]) if not pd.isna(coef_df.loc[hp, "ci_upper"]) else None,
            }
        simple_results["help_effects"] = help_effects

    # Create a concise textual interpretation automatically using p<0.05 threshold if p-values available
    def interpret_coef(name):
        if name not in coef_df.index:
            return None
        row = coef_df.loc[name]
        p = row["p"]
        coef = row["coef"]
        sig = None
        if not pd.isna(p):
            sig = (p < 0.05)
        direction = "positive" if coef > 0 else ("negative" if coef < 0 else "no effect")
        s = f"Parameter '{name}': coefficient = {coef:.4g}"
        if not pd.isna(row["se"]):
            s += f", SE = {row['se']:.4g}"
        if not pd.isna(p):
            s += f", p = {p:.3g}"
        if sig is not None:
            if sig:
                s += f" -> statistically significant ({direction})."
            else:
                s += " -> not statistically significant."
        return s

    interpretations = []
    # Summarize age main
    if age_param:
        interpretations.append(interpret_coef(age_param))
        if age_help_param:
            interpretations.append(interpret_coef(age_help_param))
            # interpret simple slopes if computed
            if "age_when_help_contrast" in simple_results:
                ar = simple_results["age_when_help_contrast"]
                p = ar.get("p")
                sig = (p is not None and not np.isnan(p) and p < 0.05)
                interpretations.append(
                    f"Effect of age when receiving help (age + interaction): est = {ar['est']:.4g},"
                    + (f" p = {ar['p']:.3g}" if ar.get("p") is not None else "")
                    + (f" -> {'significant' if sig else 'not significant'}." if p is not None else ".")
                )
    # Summarize sex main(s)
    if sex_params:
        for sp in sex_params:
            interpretations.append(interpret_coef(sp))
            # if computed conditional effect when help present
            key = sp + "_when_help_contrast"
            if "sex_effects" in simple_results and key in simple_results["sex_effects".replace("","")]:
                # unlikely path; handled below by sex_effects contents
                pass
        # add any computed sex conditional effects
        for k, v in simple_results.get("sex_effects", {}).items():
            # skip the base entries (which we already added)
            if isinstance(v, dict) and "est" in v and "combination" in v:
                p = v.get("p")
                sig = (p is not None and not np.isnan(p) and p < 0.05)
                interpretations.append(
                    f"Effect for {k} when help = contrasted level: est = {v['est']:.4g}"
                    + (f", p = {v['p']:.3g}" if v.get("p") is not None else "")
                    + (f" -> {'significant' if sig else 'not significant'}." if p is not None else ".")
                )

    # Summarize help main(s)
    if help_params:
        for hp in help_params:
            interpretations.append(interpret_coef(hp))

    # Final short human-friendly summary line
    # Use significance of main terms and interactions if available to produce a concise statement
    summary_lines = []
    # Age summary
    if age_param:
        p = coef_df.loc[age_param, "p"]
        coef = coef_df.loc[age_param, "coef"]
        if not pd.isna(p) and p < 0.05:
            summary_lines.append(f"Age is associated with efficiency: each additional year changes nuts/min by {coef:.3g} (p={p:.3g}).")
        else:
            summary_lines.append("No clear evidence that age alone is associated with efficiency (age term not statistically significant).")
    # Help x age interaction
    if age_help_param:
        p = coef_df.loc[age_help_param, "p"]
        coef = coef_df.loc[age_help_param, "coef"]
        if not pd.isna(p) and p < 0.05:
            summary_lines.append(f"The effect of age differs when help is received (interaction coef = {coef:.3g}, p={p:.3g}).")
            if "age_when_help_contrast" in simple_results:
                ar = simple_results["age_when_help_contrast"]
                summary_lines.append(f"Estimated age effect when help is present: {ar['est']:.3g} nuts/min per year (p={ar['p']:.3g} if available).")
        else:
            summary_lines.append("No clear evidence that the age effect differs by help (age:help interaction not statistically significant).")
    # Sex summary
    if sex_params:
        # if multiple contrasts, just report generically
        any_sig = any((not pd.isna(coef_df.loc[s, "p"]) and coef_df.loc[s, "p"] < 0.05) for s in sex_params)
        if any_sig:
            summary_lines.append("Sex (contrast) has at least one statistically significant association with efficiency.")
        else:
            summary_lines.append("No clear evidence that sex alone is associated with efficiency (sex contrasts not statistically significant).")
    # Help main
    if help_params:
        any_sig = any((not pd.isna(coef_df.loc[h, "p"]) and coef_df.loc[h, "p"] < 0.05) for h in help_params)
        if any_sig:
            summary_lines.append("Receiving help is associated with a difference in efficiency (help contrast significant).")
        else:
            summary_lines.append("No clear evidence that receiving help alone is associated with efficiency (help contrast not statistically significant).")

    # Build output object
    out_obj = {
        "coef_table": coef_df.to_dict(orient="index"),
        "simple_results": simple_results,
        # include a compact list of the key interpretation strings
        "interpretations": interpretations,
        "summary_lines": summary_lines,
    }

    # Build a human-readable description summarizing what the object contains
    description = (
        "Extracted coefficient table (estimates, SEs, p-values, 95% CIs when available) for all fixed effects, "
        "plus computed simple effects:\n"
        "- Age main effect and age:help simple slope (age effect when help is present) if the interaction exists.\n"
        "- Sex contrasts (reported for whichever level(s) were coded) and sex:help combined effects if available.\n"
        "- Help contrasts (reported for whichever level(s) were coded).\n\n"
        "Also provides concise interpretation lines and a short summary indicating whether age, sex, help, or the interactions "
        "appear statistically significant (using p < 0.05 when p-values are available). "
        "Use the 'coef_table' for exact coefficient values and 'simple_results' for conditional/interaction summaries."
    )

    return {"object": out_obj, "description": description}