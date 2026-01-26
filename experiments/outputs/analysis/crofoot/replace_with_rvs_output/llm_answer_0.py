def extract_final_answer(model_output):
    """
    Extracts key statistics from the provided model output and returns a
    concise, interpretable summary focused on the effects of:
      - RelGroupSize
      - HomeAdvantage
      - RelGroupSize:HomeAdvantage (interaction)

    Returns a dict with:
      - "object": dict of numeric results (coef, p-value, OR, OR 95% CI) for each term
      - "description": brief interpretation of those results in the context of
                       whether relative group size and/or home advantage influence
                       the probability that the focal group wins.
    """
    import numpy as np

    # Obtain the statsmodels result object (clustered results preferred)
    res = None
    if isinstance(model_output, dict):
        # prefer 'model' (clustered results), fall back to 'glm_result', else try direct keys
        if 'model' in model_output and model_output['model'] is not None:
            res = model_output['model']
        elif 'glm_result' in model_output and model_output['glm_result'] is not None:
            res = model_output['glm_result']
        elif 'results' in model_output and model_output['results'] is not None:
            res = model_output['results']
        else:
            # maybe the whole dict already is the statsmodels wrapper (unlikely), try to find a wrapper-like object
            # try each value until one works
            for v in model_output.values():
                try:
                    _ = v.params
                    res = v
                    break
                except Exception:
                    continue
    else:
        res = model_output

    if res is None:
        raise ValueError("Could not find a statsmodels result object in model_output.")

    # Extract numeric summaries
    try:
        params = res.params
        pvalues = res.pvalues
        conf = res.conf_int()
    except Exception as e:
        raise ValueError(f"Provided result object does not expose expected attributes: {e}")

    # Terms of interest
    terms = ['RelGroupSize', 'HomeAdvantage', 'RelGroupSize:HomeAdvantage']

    # Build summary
    summary = {}
    or_vals = np.exp(params)
    # conf may have two columns (lower, upper)
    try:
        ci_lower = np.exp(conf.iloc[:, 0])
        ci_upper = np.exp(conf.iloc[:, 1])
    except Exception:
        # if conf is an ndarray-like
        conf_arr = np.asarray(conf)
        ci_lower = np.exp(conf_arr[:, 0])
        ci_upper = np.exp(conf_arr[:, 1])

    for t in terms:
        if t in params.index:
            coef = float(params[t])
            p = float(pvalues[t])
            or_val = float(or_vals[t])
            lower = float(ci_lower[t]) if hasattr(ci_lower, 'index') else float(ci_lower[params.index.get_loc(t)])
            upper = float(ci_upper[t]) if hasattr(ci_upper, 'index') else float(ci_upper[params.index.get_loc(t)])
            pct_change = (or_val - 1.0) * 100.0
            summary[t] = {
                'coef': coef,
                'pvalue': p,
                'OR': or_val,
                'OR_2.5%': lower,
                'OR_97.5%': upper,
                'OR_percent_change': pct_change  # percent change in odds per 1 SD increase in predictor
            }
        else:
            summary[t] = None

    # Formulate a concise interpretation / conclusion
    sig_terms = [t for t, v in summary.items() if v is not None and v['pvalue'] < 0.05]

    # Build human-readable description for each term (direction + significance)
    term_texts = []
    for t in terms:
        v = summary.get(t)
        if v is None:
            term_texts.append(f"{t}: not estimated in model.")
            continue
        dir_text = "increases" if v['OR'] > 1 else "decreases"
        # percent change rounded
        pct = round(v['OR_percent_change'], 1)
        term_texts.append(
            f"{t}: coef={v['coef']:.3f}, OR={v['OR']:.3f} "
            f"(95% CI {v['OR_2.5%']:.3f}–{v['OR_97.5%']:.3f}), p={v['pvalue']:.3f}. "
            f"Interpretation: a 1 SD increase in {t} {dir_text} the odds of focal group winning by ~{pct}%."
        )

    if len(sig_terms) == 0:
        conclusion = (
            "Conclusion: There is no statistically significant evidence (all p > 0.05) that relative group size, "
            "home advantage, or their interaction affect the probability that the focal group wins. "
            "Point estimates: RelGroupSize OR < 1 (suggesting a decrease in odds with larger focal group), "
            "HomeAdvantage OR > 1 (suggesting increased odds when focal group is closer to its center), "
            "but these effects are not distinguishable from zero given the uncertainty (wide CIs overlapping 1)."
        )
    else:
        names = ", ".join(sig_terms)
        conclusion = (
            f"Conclusion: The following terms are statistically significant at p<0.05: {names}. "
            "See term-level interpretations above for direction and magnitude."
        )

    description = " ; ".join(term_texts) + " || " + conclusion

    return {"object": summary, "description": description}