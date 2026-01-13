def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'HasChildren' on 'Affairs' from the
    provided model_output dict (expected keys: 'zinb' and 'ols_robust').

    Returns:
      {
        "object": {
          "zinb": {
            "count_coef": float or None,         # coefficient in count model
            "count_se": float or None,
            "count_pvalue": float or None,
            "count_ci": [lower, upper] or None,  # 95% CI on coefficient
            "irr": float or None,                # exp(coef)
            "irr_ci": [lower, upper] or None,    # CI on IRR (exp of coef CI)
            "inflation_coef": float or None,     # coefficient in inflation (logit) model
            "inflation_se": float or None,
            "inflation_pvalue": float or None,
            "inflation_ci": [lower, upper] or None
          } or None,
          "ols_robust": {
            "coef": float or None,
            "se": float or None,
            "pvalue": float or None,
            "ci": [lower, upper] or None
          } or None
        },
        "description": "Brief interpretation string"
      }
    """
    import numpy as np
    import math

    out = {"zinb": None, "ols_robust": None}
    desc_lines = []

    # Helper to find parameter name in result.params index
    def find_param_name(params_index, base_name, inflate=False):
        """
        Try to find the best match for base_name in params_index.
        If inflate=True, prefer a name that contains 'inflate' or startswith 'inflate_'.
        Otherwise prefer names that do NOT contain 'inflate'.
        Returns the matched name or None.
        """
        names = list(params_index)
        # exact match first
        if base_name in names:
            # ensure inflate flag matches
            if inflate:
                # see if there is an 'inflate_' variant
                for n in names:
                    if ('inflate' in n.lower() or n.lower().startswith('inflate_')) and base_name in n:
                        return n
                # fallback to base_name if no explicit inflate found
            else:
                return base_name

        # search more loosely
        if inflate:
            for n in names:
                if ('inflate' in n.lower() or n.lower().startswith('inflate_')) and base_name.lower() in n.lower():
                    return n
        else:
            # prefer names that contain base_name but not 'inflate'
            for n in names:
                if base_name.lower() in n.lower() and 'inflate' not in n.lower():
                    return n
        # last resort: any name that endswith or contains base_name
        for n in names:
            if base_name.lower() in n.lower():
                return n
        return None

    # Process ZINB
    zinb_res = model_output.get('zinb', None)
    if zinb_res is None:
        desc_lines.append("ZINB model not provided.")
    elif isinstance(zinb_res, Exception):
        desc_lines.append(f"ZINB model failed with exception: {repr(zinb_res)}")
    else:
        try:
            params = zinb_res.params  # pandas Series-like
            bse = zinb_res.bse
            pvalues = None
            try:
                pvalues = zinb_res.pvalues
            except Exception:
                # some older versions may not have pvalues attribute; attempt t_test
                pvalues = None

            # find parameter names
            count_name = find_param_name(params.index, "HasChildren", inflate=False)
            infl_name = find_param_name(params.index, "HasChildren", inflate=True)

            zinb_obj = {}

            # Count component stats
            if count_name is not None and count_name in params.index:
                coef = float(params[count_name])
                se = float(bse[count_name]) if (bse is not None and count_name in bse.index) else None
                pval = float(pvalues[count_name]) if (pvalues is not None and count_name in pvalues.index) else None

                # Confidence interval
                try:
                    ci = zinb_res.conf_int()
                    # conf_int may be a DataFrame or ndarray; try to index by name
                    if hasattr(ci, 'loc'):
                        ci_lower, ci_upper = float(ci.loc[count_name, 0]), float(ci.loc[count_name, 1])
                    else:
                        # fallback: use coef +/- 1.96*se
                        if se is not None:
                            ci_lower = coef - 1.96 * se
                            ci_upper = coef + 1.96 * se
                        else:
                            ci_lower = ci_upper = None
                except Exception:
                    if se is not None:
                        ci_lower = coef - 1.96 * se
                        ci_upper = coef + 1.96 * se
                    else:
                        ci_lower = ci_upper = None

                # IRR and its CI by exponentiating coef and CI
                try:
                    irr = float(np.exp(coef))
                    irr_ci = [float(np.exp(ci_lower)) if ci_lower is not None else None,
                              float(np.exp(ci_upper)) if ci_upper is not None else None]
                except Exception:
                    irr = None
                    irr_ci = None

                zinb_obj.update({
                    "count_coef": coef,
                    "count_se": se,
                    "count_pvalue": pval,
                    "count_ci": [ci_lower, ci_upper],
                    "irr": irr,
                    "irr_ci": irr_ci
                })
            else:
                zinb_obj.update({
                    "count_coef": None,
                    "count_se": None,
                    "count_pvalue": None,
                    "count_ci": None,
                    "irr": None,
                    "irr_ci": None
                })

            # Inflation (logit) component stats
            if infl_name is not None and infl_name in params.index:
                ic = float(params[infl_name])
                ise = float(bse[infl_name]) if (bse is not None and infl_name in bse.index) else None
                ipval = float(pvalues[infl_name]) if (pvalues is not None and infl_name in pvalues.index) else None

                try:
                    ci = zinb_res.conf_int()
                    if hasattr(ci, 'loc'):
                        infl_lower, infl_upper = float(ci.loc[infl_name, 0]), float(ci.loc[infl_name, 1])
                    else:
                        if ise is not None:
                            infl_lower = ic - 1.96 * ise
                            infl_upper = ic + 1.96 * ise
                        else:
                            infl_lower = infl_upper = None
                except Exception:
                    if ise is not None:
                        infl_lower = ic - 1.96 * ise
                        infl_upper = ic + 1.96 * ise
                    else:
                        infl_lower = infl_upper = None

                zinb_obj.update({
                    "inflation_coef": ic,
                    "inflation_se": ise,
                    "inflation_pvalue": ipval,
                    "inflation_ci": [infl_lower, infl_upper]
                })
            else:
                zinb_obj.update({
                    "inflation_coef": None,
                    "inflation_se": None,
                    "inflation_pvalue": None,
                    "inflation_ci": None
                })

            out["zinb"] = zinb_obj

            # Add short interpretive sentence for ZINB
            if zinb_obj["count_coef"] is not None:
                sign = "decrease" if zinb_obj["count_coef"] < 0 else "increase"
                pstr = f"p={zinb_obj['count_pvalue']:.3f}" if zinb_obj['count_pvalue'] is not None else "p=NA"
                irr = zinb_obj.get("irr", None)
                if irr is not None:
                    pct = (1 - irr) * 100 if irr < 1 else (irr - 1) * 100
                    pct_str = f"{abs(pct):.1f}% {'lower' if irr<1 else 'higher'} expected affairs"
                else:
                    pct_str = ""
                desc_lines.append(
                    f"ZINB count: HasChildren coef = {zinb_obj['count_coef']:.4f} ({pstr}), "
                    f"IRR = {irr:.3f} ({pct_str}). This indicates a {sign} in the expected count of affairs "
                    f"associated with having children (count model)."
                )
            else:
                desc_lines.append("ZINB count: no 'HasChildren' coefficient found.")

            # Inflation interpretation
            if zinb_obj["inflation_coef"] is not None:
                ip = zinb_obj["inflation_pvalue"]
                pstr = f"p={ip:.3f}" if ip is not None else "p=NA"
                desc_lines.append(
                    f"ZINB inflation (logit) component: HasChildren coef = {zinb_obj['inflation_coef']:.4f} ({pstr}); "
                    "positive values mean higher log-odds of being an 'always-zero' (no-affair) case."
                )
        except Exception as e:
            desc_lines.append(f"Failed to extract ZINB statistics: {repr(e)}")

    # Process OLS robust
    ols_res = model_output.get('ols_robust', None)
    if ols_res is None:
        desc_lines.append("OLS model not provided.")
    elif isinstance(ols_res, Exception):
        desc_lines.append(f"OLS model failed with exception: {repr(ols_res)}")
    else:
        try:
            params = ols_res.params
            bse = ols_res.bse
            pvalues = ols_res.pvalues
            # confidence intervals
            try:
                ci_df = ols_res.conf_int()
            except Exception:
                ci_df = None

            if "HasChildren" in params.index:
                coef = float(params["HasChildren"])
                se = float(bse["HasChildren"]) if "HasChildren" in bse.index else None
                pval = float(pvalues["HasChildren"]) if "HasChildren" in pvalues.index else None
                if ci_df is not None and hasattr(ci_df, 'loc'):
                    ci_lower, ci_upper = float(ci_df.loc["HasChildren", 0]), float(ci_df.loc["HasChildren", 1])
                else:
                    if se is not None:
                        ci_lower = coef - 1.96 * se
                        ci_upper = coef + 1.96 * se
                    else:
                        ci_lower = ci_upper = None

                out["ols_robust"] = {
                    "coef": coef,
                    "se": se,
                    "pvalue": pval,
                    "ci": [ci_lower, ci_upper]
                }

                pstr = f"p={pval:.3f}" if pval is not None else "p=NA"
                sign = "decrease" if coef < 0 else "increase"
                desc_lines.append(
                    f"OLS (HC3): HasChildren coef = {coef:.4f} ({pstr}); this is an estimated {sign} in reported "
                    "affairs associated with having children (linear approximation)."
                )
            else:
                out["ols_robust"] = None
                desc_lines.append("OLS: no 'HasChildren' coefficient found.")
        except Exception as e:
            desc_lines.append(f"Failed to extract OLS statistics: {repr(e)}")

    # Provide an overall short conclusion based on signs and statistical significance (simple heuristic)
    concl = ""
    try:
        z = out.get("zinb")
        o = out.get("ols_robust")
        zinb_sig = False
        ols_sig = False
        zinb_dir = None
        ols_dir = None
        if z:
            if z["count_pvalue"] is not None and z["count_pvalue"] < 0.05:
                zinb_sig = True
            if z["count_coef"] is not None:
                zinb_dir = "negative" if z["count_coef"] < 0 else "positive"
        if o:
            if o["pvalue"] is not None and o["pvalue"] < 0.05:
                ols_sig = True
            if o["coef"] is not None:
                ols_dir = "negative" if o["coef"] < 0 else "positive"

        if zinb_sig and zinb_dir == "negative":
            concl = "Primary ZINB result: having children is associated with a statistically significant decrease in expected number of affairs."
        elif zinb_sig and zinb_dir == "positive":
            concl = "Primary ZINB result: having children is associated with a statistically significant increase in expected number of affairs."
        else:
            # if ZINB not significant, check OLS
            if ols_sig and ols_dir == "negative":
                concl = "OLS robustness: having children is associated with a statistically significant decrease in reported affairs (linear model)."
            elif ols_sig and ols_dir == "positive":
                concl = "OLS robustness: having children is associated with a statistically significant increase in reported affairs (linear model)."
            else:
                concl = "No strong evidence of a statistically significant association between having children and engagement in extramarital affairs based on these models."

        desc_lines.append(concl)
    except Exception:
        # fallback if anything goes wrong
        desc_lines.append("Could not form an overall conclusion programmatically.")

    return {"object": out, "description": " ".join(desc_lines)}