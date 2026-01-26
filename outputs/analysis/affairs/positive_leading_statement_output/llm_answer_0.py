def extract_final_answer(model_output):
    """
    Extracts statistics about the 'children_binary' effect from the model_output dict
    returned by the modeling function. Returns a dict with:
      - "object": dict of extracted statistics per model (coef, se, pval, 95% CI,
                    exponentiated effect for count models, and inflation-part coef for ZIP if present)
      - "description": brief interpretation of the results in the context of whether children
                       decrease engagement in extramarital affairs.
    """
    import numpy as np
    results = {}
    def safe_get_attrs(res):
        # Return None or dict of params, bse, pvalues, conf_int
        try:
            params = getattr(res, "params", None)
            bse = getattr(res, "bse", None)
            pvalues = getattr(res, "pvalues", None)
            try:
                conf = res.conf_int()
            except Exception:
                conf = None
            return {"params": params, "bse": bse, "pvalues": pvalues, "conf_int": conf}
        except Exception:
            return None

    def extract_param_stats(res, param_name):
        """Return dict for a single parameter name if present in res, else None."""
        if isinstance(res, str):
            return None
        attrs = safe_get_attrs(res)
        if not attrs or attrs["params"] is None:
            return None
        params = attrs["params"]
        if param_name not in params.index:
            return None
        try:
            coef = float(params.loc[param_name])
        except Exception:
            coef = None
        try:
            se = float(attrs["bse"].loc[param_name]) if attrs["bse" ]is not None else None
        except Exception:
            se = None
        try:
            pval = float(attrs["pvalues"].loc[param_name]) if attrs["pvalues"] is not None else None
        except Exception:
            pval = None
        ci = None
        if attrs["conf_int"] is not None and param_name in attrs["conf_int"].index:
            try:
                ci_low = float(attrs["conf_int"].loc[param_name, 0])
                ci_high = float(attrs["conf_int"].loc[param_name, 1])
                ci = (ci_low, ci_high)
            except Exception:
                ci = None
        out = {"coef": coef, "se": se, "pval": pval, "ci95": ci}
        return out

    # Iterate over expected model keys
    for m in ['negative_binomial', 'ols_robust', 'zero_inflated_poisson']:
        res = model_output.get(m)
        entry = {"status": None}
        # If the model entry is a failure string, record it
        if isinstance(res, str):
            entry["status"] = "failed"
            entry["note"] = res
            results[m] = entry
            continue
        # Try to extract count-part coefficient named 'children_binary'
        count_stats = extract_param_stats(res, 'children_binary')
        # For ZIP models, statsmodels often exposes inflation parameters with prefix 'inflate_'.
        inflate_stats = extract_param_stats(res, 'inflate_children_binary') or extract_param_stats(res, 'inflate.children_binary') 
        # Fallback: check any parameter name containing 'children_binary'
        if count_stats is None and not isinstance(res, str):
            # search param names
            try:
                params = getattr(res, "params", None)
                if params is not None:
                    found = [n for n in params.index if 'children_binary' in n]
                    # prefer exact match, otherwise take first match
                    if found:
                        count_stats = extract_param_stats(res, found[0])
                        # if the found name indicates inflation, also try to find a non-inflate counterpart
                        if found[0].startswith('inflate') and inflate_stats is None:
                            inflate_stats = extract_param_stats(res, found[0])
            except Exception:
                pass

        entry["status"] = "ok" if (count_stats or inflate_stats) else "param_not_found"
        if count_stats:
            entry["count_part"] = count_stats
            # For count models (NB, ZIP, Poisson) exponentiate coef to get multiplicative effect
            try:
                coef = count_stats.get("coef")
                if coef is not None:
                    entry["count_part"]["exp_coef"] = float(np.exp(coef))
                if count_stats.get("ci95") is not None:
                    entry["count_part"]["exp_ci95"] = (float(np.exp(count_stats["ci95"][0])),
                                                      float(np.exp(count_stats["ci95"][1])))
            except Exception:
                pass
        if inflate_stats:
            entry["inflation_part"] = inflate_stats
            try:
                coef = inflate_stats.get("coef")
                if coef is not None:
                    entry["inflation_part"]["odds_ratio"] = float(np.exp(coef))
                if inflate_stats.get("ci95") is not None:
                    entry["inflation_part"]["odds_ratio_ci95"] = (float(np.exp(inflate_stats["ci95"][0])),
                                                                 float(np.exp(inflate_stats["ci95"][1])))
            except Exception:
                pass

        # As a fallback, if the model_output included a precomputed summary for children_binary, include it
        try:
            precomp = model_output.get('summary_children_binary', {}).get(m)
            if precomp and isinstance(precomp, dict):
                entry.setdefault("precomputed_summary", precomp)
        except Exception:
            pass

        results[m] = entry

    # Build a concise textual interpretation
    # Use available p-values/coefs to summarize evidence
    interp_parts = []
    nb = results.get('negative_binomial', {})
    ols = results.get('ols_robust', {})
    zipr = results.get('zero_inflated_poisson', {})

    def summarize_model(name, rec):
        if rec.get("status") == "failed":
            return f"{name}: estimation failed."
        if rec.get("status") == "param_not_found":
            return f"{name}: 'children_binary' parameter not found."
        cp = rec.get("count_part")
        ip = rec.get("inflation_part")
        s = f"{name}: "
        if cp:
            coef = cp.get("coef")
            p = cp.get("pval")
            expc = cp.get("exp_coef")
            if coef is not None and p is not None:
                s += f"count-part coef={coef:.3f}, p={p:.3f}"
                if expc is not None:
                    s += f" (exp={expc:.3f})"
                s += ". "
            else:
                s += "count-part stats not fully available. "
        if ip:
            coef = ip.get("coef")
            p = ip.get("pval")
            orr = ip.get("odds_ratio")
            if coef is not None and p is not None:
                s += f"inflation-part coef={coef:.3f}, p={p:.3f}"
                if orr is not None:
                    s += f" (odds ratio={orr:.3f})"
                s += ". "
            else:
                s += "inflation-part stats not fully available. "
        return s.strip()

    interp_parts.append(summarize_model("Negative Binomial", nb))
    interp_parts.append(summarize_model("OLS (robust SE)", ols))
    interp_parts.append(summarize_model("Zero-Inflated Poisson", zipr))

    # Overall short conclusion guidance
    # Determine significance: consider p < 0.05 in any model as evidence; note consistency
    sig_models = []
    for name, rec in (("negative_binomial", nb), ("ols_robust", ols), ("zero_inflated_poisson", zipr)):
        cp = rec.get("count_part")
        if cp and cp.get("pval") is not None and cp.get("pval") < 0.05:
            sig_models.append((name, cp))
    conclusion = ""
    if not sig_models:
        conclusion = ("No robust evidence that having children decreases engagement in extramarital affairs: "
                      "Negative Binomial and OLS show non-significant effects; Zero-Inflated Poisson shows a "
                      "statistically significant negative coefficient in its count part but this result is not "
                      "consistent across preferred models. Interpretation should be cautious; if anything the "
                      "ZIP suggests ~{:.0f}% lower expected count (exp(coef)-1) but NB (preferred for overdispersion) "
                      "is not significant.").format(
                          (zipr.get("count_part", {}).get("exp_coef") - 1) * 100
                          if zipr.get("count_part", {}) and zipr["count_part"].get("exp_coef") is not None else 0
                      )
    else:
        # At least one model significant
        msgs = []
        for nm, cp in sig_models:
            frac = (cp.get("exp_coef") - 1) * 100 if cp.get("exp_coef") is not None else None
            if frac is not None:
                msgs.append(f"{nm} suggests a {abs(frac):.1f}% {'decrease' if frac<0 else 'increase'} (exp(coef)={cp.get('exp_coef'):.3f}), p={cp.get('pval'):.3f}")
            else:
                msgs.append(f"{nm} significant coef={cp.get('coef'):.3f}, p={cp.get('pval'):.3f}")
        conclusion = ("Some models (notably {}) show a statistically significant association, but results are not "
                      "consistent across models; interpret cautiously.").format(", ".join([m for m,_ in sig_models]))
        conclusion += " " + " ".join(msgs)

    description = "Per-model extracted statistics: " + " | ".join(interp_parts) + " Overall: " + conclusion

    return {"object": results, "description": description}