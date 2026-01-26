def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of HasChildren on extramarital affair outcomes
    from the model_output produced by the provided modeling function.

    Returns a dictionary with:
      - "object": a dict containing extracted numeric results (logit and OLS results and the summary table)
      - "description": a plain-language summary interpreting those results in context
    """
    import numpy as np
    import pandas as pd

    out = {"logit": None, "ols": None, "summary_table": None}
    messages = []

    # Helper to format floats
    def fmt(x, nd=3):
        try:
            return float(np.round(x, nd))
        except Exception:
            return x

    # 1) Extract summary table if present
    summary = model_output.get("summary_table", None)
    if summary is not None:
        # If it's a DataFrame already, convert to records for portability
        if isinstance(summary, pd.DataFrame):
            out["summary_table"] = summary.to_dict(orient="records")
        else:
            # assume it's serializable (e.g., list of dicts)
            out["summary_table"] = summary
    else:
        messages.append("No summary_table found in model_output.")

    # 2) Extract logistic regression results (AnyAffair ~ HasChildren + controls)
    logit_res = model_output.get("logit", None)
    if logit_res is None:
        out["logit"] = {"message": model_output.get("logit_message", "Logistic model not present.")}
    else:
        try:
            # coefficient, se, pvalue, conf_int
            coef = float(logit_res.params["HasChildren"])
            se = float(logit_res.bse["HasChildren"]) if "bse" in dir(logit_res) else None
            pval = float(logit_res.pvalues["HasChildren"])
            ci = logit_res.conf_int().loc["HasChildren"].tolist()  # [lower, upper]
            ci = [float(ci[0]), float(ci[1])]
            # odds ratio and CI
            orr = float(np.exp(coef))
            ci_or = [float(np.exp(ci[0])), float(np.exp(ci[1]))]

            out["logit"] = {
                "coef": fmt(coef, 4),
                "se": fmt(se, 4) if se is not None else None,
                "pvalue": fmt(pval, 4),
                "conf_int_coef": [fmt(ci[0], 4), fmt(ci[1], 4)],
                "odds_ratio": fmt(orr, 4),
                "conf_int_or": [fmt(ci_or[0], 4), fmt(ci_or[1], 4)],
                "nobs": int(getattr(logit_res, "nobs", np.nan))
            }
        except Exception as e:
            out["logit"] = {"message": f"Failed to extract logit stats: {e}"}

    # 3) Extract OLS results (LogAffairFreqPos ~ HasChildren + controls), conditional on positive affairs
    ols_res = model_output.get("ols", None)
    if ols_res is None:
        out["ols"] = {"message": model_output.get("ols_message", "OLS model not present.")}
    else:
        try:
            coef = float(ols_res.params["HasChildren"])
            se = float(ols_res.bse["HasChildren"]) if "bse" in dir(ols_res) else None
            pval = float(ols_res.pvalues["HasChildren"])
            ci = ols_res.conf_int().loc["HasChildren"].tolist()
            ci = [float(ci[0]), float(ci[1])]
            # Interpret coefficient on log outcome as percent change: (exp(beta)-1)*100
            pct_change = (np.exp(coef) - 1.0) * 100.0
            pct_ci = [(np.exp(ci[0]) - 1.0) * 100.0, (np.exp(ci[1]) - 1.0) * 100.0]

            out["ols"] = {
                "coef_log": fmt(coef, 4),
                "se": fmt(se, 4) if se is not None else None,
                "pvalue": fmt(pval, 4),
                "conf_int_coef": [fmt(ci[0], 4), fmt(ci[1], 4)],
                "percent_change_associated": fmt(pct_change, 2),
                "percent_change_conf_int": [fmt(pct_ci[0], 2), fmt(pct_ci[1], 2)],
                "nobs": int(getattr(ols_res, "nobs", np.nan))
            }
        except Exception as e:
            out["ols"] = {"message": f"Failed to extract OLS stats: {e}"}

    # 4) Compose a plain-language interpretation
    desc_lines = []
    # Add raw summary means if available
    if out["summary_table"]:
        # Expect two rows: HasChildren == 0 and 1
        try:
            rows = {int(r["HasChildren"]): r for r in out["summary_table"]}
            no_kids = rows.get(0)
            kids = rows.get(1)
            if no_kids and kids:
                desc_lines.append(
                    f"Raw means: Any-affair rate without children = {fmt(no_kids['AnyAffairRate'],3)} "
                    f"(N={int(no_kids['N'])}), with children = {fmt(kids['AnyAffairRate'],3)} (N={int(kids['N'])})."
                )
                desc_lines.append(
                    f"Mean affair frequency: without children = {fmt(no_kids['MeanAffairFreq'],3)}, "
                    f"with children = {fmt(kids['MeanAffairFreq'],3)}."
                )
        except Exception:
            # fallback: just include the summary_table as-is
            desc_lines.append("Summary table present; see numeric output for group means.")

    # Interpret logit
    if isinstance(out["logit"], dict) and "coef" in out["logit"]:
        l = out["logit"]
        sig_text = "statistically significant" if (l["pvalue"] < 0.05) else "not statistically significant"
        desc_lines.append(
            f"Logistic regression (AnyAffair): HasChildren coef = {l['coef']} (SE={l['se']}); "
            f"odds ratio = {l['odds_ratio']} (95% CI [{l['conf_int_or'][0]}, {l['conf_int_or'][1]}]), "
            f"p = {l['pvalue']}. This means having children is associated with {'' if l['odds_ratio']>=1 else 'lower '}odds "
            f"of reporting any affair; the effect is {sig_text}."
        )
    else:
        desc_lines.append(f"Logistic model not available or no extractable results. {out['logit'].get('message','')}")

    # Interpret OLS
    if isinstance(out["ols"], dict) and "coef_log" in out["ols"]:
        o = out["ols"]
        sig_text = "statistically significant" if (o["pvalue"] < 0.05) else "not statistically significant"
        desc_lines.append(
            f"Conditional OLS (among those with any affair, log-frequency): HasChildren coef = {o['coef_log']} (SE={o['se']}), "
            f"which corresponds to an associated change of about {o['percent_change_associated']}% "
            f"in affair frequency (95% CI [{o['percent_change_conf_int'][0]}%, {o['percent_change_conf_int'][1]}%]), "
            f"p = {o['pvalue']}. The effect is {sig_text}."
        )
    else:
        desc_lines.append(f"OLS model not available or no extractable results. {out['ols'].get('message','')}")

    # Overall conclusion (conservative)
    # We avoid causal language; state "associated"
    try:
        # Decide directional summary based on odds ratio if available
        if isinstance(out["logit"], dict) and "odds_ratio" in out["logit"]:
            orr = out["logit"]["odds_ratio"]
            p = out["logit"]["pvalue"]
            if orr > 1:
                if p < 0.05:
                    concl = "Having children is associated with higher likelihood of reporting any extramarital affair (statistically significant)."
                else:
                    concl = "Having children is associated with higher likelihood of reporting any extramarital affair (not statistically significant)."
            elif orr < 1:
                if p < 0.05:
                    concl = "Having children is associated with lower likelihood of reporting any extramarital affair (statistically significant)."
                else:
                    concl = "Having children is associated with lower likelihood of reporting any extramarital affair (not statistically significant)."
            else:
                concl = "No association between having children and the likelihood of reporting any extramarital affair was detected."
            desc_lines.append(concl)
    except Exception:
        pass

    description = " ".join(desc_lines)

    return {"object": out, "description": description}