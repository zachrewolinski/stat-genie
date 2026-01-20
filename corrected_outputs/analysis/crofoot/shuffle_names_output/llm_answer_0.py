def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLMResultsWrapper (logistic regression)
    and produce a concise, interpretable summary about whether relative group size and
    contest location affect the probability that the focal group wins.

    Returns:
      {
        "object": pandas.DataFrame with rows for relevant terms and columns:
                    ['coef','se','pval','conf_low','conf_high',
                     'odds_ratio','or_conf_low','or_conf_high','significant','direction']
        "description": short human-readable interpretation string
      }
    """
    import numpy as np
    import pandas as pd

    def _truthy(val):
        """Robust truth check: handle scalars, pandas Series, numpy arrays.
        Returns True if any element is truthy; False for NA/empty/False."""
        try:
            if val is None:
                return False
            # pandas NA
            if isinstance(val, (pd.Series, pd.DataFrame)):
                if val.size == 0:
                    return False
                # If DataFrame, reduce to values
                arr = val.values.ravel()
            elif isinstance(val, np.ndarray):
                if val.size == 0:
                    return False
                arr = val.ravel()
            else:
                # scalar
                return bool(val)
            # For array-like, consider True if any element is truthy (and not NA)
            # Convert to boolean, ignoring NA
            bools = []
            for x in arr:
                try:
                    if pd.isna(x):
                        continue
                except Exception:
                    pass
                try:
                    if bool(x):
                        return True
                except Exception:
                    # fallback: check numeric nonzero
                    try:
                        if float(x) != 0:
                            return True
                    except Exception:
                        continue
            return False
        except Exception:
            # conservative fallback
            try:
                return bool(val)
            except Exception:
                return False

    # get params, pvalues, se, conf_int
    try:
        params = model_output.params
        pvals = model_output.pvalues
        bse = model_output.bse
        conf = model_output.conf_int()  # DataFrame or ndarray with 2 columns
        # Ensure conf has columns we expect
        if isinstance(conf, np.ndarray):
            conf = pd.DataFrame(conf, index=params.index, columns=['conf_low', 'conf_high'])
        else:
            # Ensure consistent column names regardless of what statsmodels returns
            conf = conf.copy()
            conf.columns = ['conf_low', 'conf_high']
    except Exception as e:
        raise ValueError(f"Provided model_output does not look like a statsmodels results object: {e}")

    # Identify terms of interest
    idx = list(params.index)

    # Candidate names we expect from the model
    candidates = []
    # relative size main effects
    for name in ['size_diff', 'size_ratio']:
        if name in idx:
            candidates.append(name)
    # location dummies (any param starting with 'Location_')
    location_terms = [n for n in idx if n.startswith('Location_')]
    candidates.extend(location_terms)
    # interactions (anything containing '_x_size_diff')
    interaction_terms = [n for n in idx if '_x_size_diff' in n]
    candidates.extend(interaction_terms)

    # If nothing found, include at least size_diff if present else return helpful message
    if len(candidates) == 0:
        # nothing matched; include all params so user can inspect
        candidates = idx

    # Build results table
    rows = []
    for name in candidates:
        coef = float(params.loc[name])
        se = float(bse.loc[name]) if (hasattr(bse, "index") and name in bse.index) else np.nan
        pval = float(pvals.loc[name]) if (hasattr(pvals, "index") and name in pvals.index) else np.nan
        try:
            conf_low = float(conf.loc[name, 'conf_low'])
            conf_high = float(conf.loc[name, 'conf_high'])
        except Exception:
            # fallback if conf is ndarray-like or missing
            conf_low = np.nan
            conf_high = np.nan
        or_val = float(np.exp(coef))
        or_low = float(np.exp(conf_low)) if not np.isnan(conf_low) else np.nan
        or_high = float(np.exp(conf_high)) if not np.isnan(conf_high) else np.nan
        try:
            significant = bool((not np.isnan(pval)) and (pval < 0.05))
        except Exception:
            significant = False
        direction = 'positive' if coef > 0 else ('negative' if coef < 0 else 'zero')

        rows.append({
            'term': name,
            'coef': coef,
            'se': se,
            'pval': pval,
            'conf_low': conf_low,
            'conf_high': conf_high,
            'odds_ratio': or_val,
            'or_conf_low': or_low,
            'or_conf_high': or_high,
            'significant': significant,
            'direction': direction
        })

    results_table = pd.DataFrame(rows).set_index('term')

    # Construct a human-readable interpretation
    lines = []
    # Interpret relative size
    if 'size_diff' in results_table.index:
        r = results_table.loc['size_diff']
        sig = _truthy(results_table.loc['size_diff', 'significant'])
        if sig:
            lines.append(
                f"Relative group size (size_diff) has a statistically significant effect on win probability "
                f"(coef={r['coef']:.3f}, p={r['pval']:.3g}). "
                f"Each one-unit increase in size_diff multiplies the odds of the focal group winning by "
                f"{r['odds_ratio']:.3f} (95% CI: {r['or_conf_low']:.3f}–{r['or_conf_high']:.3f}). "
                f"Direction: {r['direction']} effect."
            )
        else:
            lines.append(
                f"Relative group size (size_diff) is estimated with coef={r['coef']:.3f} (p={r['pval']:.3g}); "
                f"this is not statistically significant at alpha=0.05, so we do not have strong evidence that "
                f"size_diff alone affects win probability."
            )
    elif 'size_ratio' in results_table.index:
        r = results_table.loc['size_ratio']
        sig = _truthy(results_table.loc['size_ratio', 'significant'])
        if sig:
            lines.append(
                f"Relative group size (size_ratio) has a statistically significant effect (coef={r['coef']:.3f}, p={r['pval']:.3g}). "
                f"Interpretation: a unit increase in size_ratio multiplies the odds of focal group winning by "
                f"{r['odds_ratio']:.3f} (95% CI: {r['or_conf_low']:.3f}–{r['or_conf_high']:.3f}). Direction: {r['direction']}."
            )
        else:
            lines.append(
                f"Relative group size (size_ratio) has coef={r['coef']:.3f} (p={r['pval']:.3g}) and is not statistically significant."
            )
    else:
        lines.append("No explicit size_diff or size_ratio term found in the model results returned.")

    # Interpret interactions (moderation by location)
    significant_interactions = []
    for name in interaction_terms:
        if name in results_table.index:
            is_sig = _truthy(results_table.loc[name, 'significant'])
            if is_sig:
                r = results_table.loc[name]
                significant_interactions.append((name, r))
    if len(significant_interactions) > 0:
        s = "; ".join([f"{n} (coef={r['coef']:.3f}, p={r['pval']:.3g})" for n, r in significant_interactions])
        lines.append(
            "There is evidence that contest location moderates the effect of relative group size: "
            f"significant interaction term(s): {s}. This means the effect of size_diff on win probability "
            "differs depending on location."
        )
    else:
        # also check whether any interaction terms were present but none significant
        present_interactions = [n for n in interaction_terms if n in results_table.index]
        if len(present_interactions) > 0:
            lines.append(
                "Interaction terms between size_diff and Location were included but none reached statistical significance "
                "at alpha=0.05, suggesting no clear evidence that location strongly moderates the size_diff effect."
            )
        else:
            lines.append("No interaction terms between size_diff and location were present in the fitted model.")

    # Interpret location main effects
    sig_locs = []
    for n in location_terms:
        if n in results_table.index and _truthy(results_table.loc[n, 'significant']):
            sig_locs.append(n)

    if len(sig_locs) > 0:
        s = ", ".join([f"{n} (coef={results_table.at[n,'coef']:.3f}, p={results_table.at[n,'pval']:.3g})" for n in sig_locs])
        lines.append(f"Some location main effects are significant: {s}. This indicates baseline differences in win odds by location.")
    else:
        if len(location_terms) > 0:
            lines.append("Location dummies were included but none had statistically significant main effects at alpha=0.05.")
        else:
            lines.append("No location dummy terms were detected in the results.")

    # Combine into description
    description = " ".join(lines)

    return {"object": results_table, "description": description}