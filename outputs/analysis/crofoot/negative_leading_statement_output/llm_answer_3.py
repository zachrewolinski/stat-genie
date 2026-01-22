def extract_final_answer(model_output):
    """
    Extract key statistics for SizeDiff_z, DistDiff_z, and their interaction from the
    model output returned by the modeling function. Returns a dict with:
      - "object": dict of extracted numeric results for each focal predictor + metadata
      - "description": a short interpretation in the context of the task
    
    The function is defensive: it detects numerical instability / complete separation
    (e.g., infinite coefficients, NaN log-likelihood) and reports that estimates are
    unreliable when that occurs.
    """
    import numpy as np

    # Prepare return structure
    out = {
        'estimates': {},
        'separation_or_instability': False,
        'notes': []
    }

    # Required variable names we care about
    vars_of_interest = ['SizeDiff_z', 'DistDiff_z', 'SizeDiff_z:DistDiff_z']

    # Get model_fit if present
    model_fit = model_output.get('model_fit', None)

    if model_fit is None:
        # Fall back to odds_ratios if model_fit missing
        or_df = model_output.get('odds_ratios', None)
        if or_df is None:
            raise ValueError("model_output does not contain 'model_fit' or 'odds_ratios'.")
        # Build minimal output from odds_ratios DataFrame
        for v in vars_of_interest:
            if v in or_df.index:
                row = or_df.loc[v]
                out['estimates'][v] = {
                    'OR': float(np.nan if np.isinf(row['OR']) and np.isnan(row['OR']) else row['OR']),
                    'CI_2.5%': float(row['2.5%']),
                    'CI_97.5%': float(row['97.5%'])
                }
            else:
                out['estimates'][v] = None
        out['separation_or_instability'] = True
        out['notes'].append("Only odds-ratio table available; numerical instability likely.")
    else:
        # Try extracting params, bse, pvalues, conf_int
        params = getattr(model_fit, 'params', None)
        bse = getattr(model_fit, 'bse', None)
        pvalues = getattr(model_fit, 'pvalues', None)
        conf = None
        try:
            conf = model_fit.conf_int()
        except Exception:
            conf = None

        # Diagnostics: check log-likelihood or extremely large coefficients
        llf = None
        try:
            llf = float(getattr(model_fit, 'llf', np.nan))
        except Exception:
            llf = np.nan

        # flag separation/instability if llf is NaN or any parameter extremely large
        unstable = False
        if np.isnan(llf):
            unstable = True
            out['notes'].append("Log-likelihood is NaN -> likely complete or quasi-complete separation or numerical failure.")
        if params is not None:
            if np.any(np.abs(params) > 1e8):
                unstable = True
                out['notes'].append("At least one coefficient has very large magnitude (>|1e8|) -> estimates likely unstable.")
        out['separation_or_instability'] = unstable

        # Extract for each variable
        for v in vars_of_interest:
            if params is None or v not in params.index:
                out['estimates'][v] = None
                continue
            coef = float(params.loc[v])
            se = float(bse.loc[v]) if (bse is not None and v in bse.index) else None
            pval = float(pvalues.loc[v]) if (pvalues is not None and v in pvalues.index) else None
            # z stat
            z = float(coef / se) if (se not in (None, 0)) else None
            # OR and CI
            try:
                OR = float(np.exp(coef)) if np.isfinite(coef) and abs(coef) < 700 else (np.inf if coef > 0 else 0.0)
            except OverflowError:
                OR = np.inf if coef > 0 else 0.0
            if conf is not None and v in conf.index:
                ci_low = conf.loc[v, 0]
                ci_high = conf.loc[v, 1]
                # exponentiate, handle overflow
                try:
                    ci_low_exp = float(np.exp(ci_low)) if np.isfinite(ci_low) and abs(ci_low) < 700 else (np.inf if ci_low > 0 else 0.0)
                    ci_high_exp = float(np.exp(ci_high)) if np.isfinite(ci_high) and abs(ci_high) < 700 else (np.inf if ci_high > 0 else 0.0)
                except OverflowError:
                    ci_low_exp = (np.inf if ci_low > 0 else 0.0)
                    ci_high_exp = (np.inf if ci_high > 0 else 0.0)
            else:
                ci_low_exp = None
                ci_high_exp = None

            out['estimates'][v] = {
                'coef': coef,
                'se': se,
                'z': z,
                'pvalue': pval,
                'OR': OR,
                'CI_2.5%': ci_low_exp,
                'CI_97.5%': ci_high_exp
            }

    # Construct a concise human-readable description/interpretation
    # If unstable, warn and provide directions; otherwise give interpretation for the three terms.
    if out['separation_or_instability']:
        desc_lines = [
            "Model estimates appear numerically unstable (NaN log-likelihood or extremely large coefficients).",
            "Reported coefficients/ORs are not reliable for inference. This commonly indicates complete or quasi-complete separation or severe collinearity (often due to many dyad fixed effects relative to observations).",
            "From the raw estimates in this fit (not reliable):",
        ]
        # Add directional glimpses if available
        for v in vars_of_interest:
            est = out['estimates'].get(v)
            if est:
                coef = est.get('coef')
                OR = est.get('OR')
                if coef is None:
                    desc_lines.append(f" - {v}: no estimate available.")
                else:
                    direction = "positive" if coef > 0 else ("zero" if coef == 0 else "negative")
                    desc_lines.append(f" - {v}: coef {coef:.3e} ({direction}), OR ≈ {OR}.")
            else:
                desc_lines.append(f" - {v}: not estimated / not present in model output.")
        desc_lines += [
            "Recommended next steps: inspect data for perfect prediction, reduce or remove dyad fixed effects (or use fewer levels), use penalized logistic regression (Firth/brglm) or a mixed-effects model, or use exact logistic regression for small samples.",
            "Do not draw substantive conclusions from these numeric values without addressing instability."
        ]
        description = " ".join(desc_lines)
    else:
        # Stable estimates: give direct interpretation
        interp = []
        for v in vars_of_interest:
            est = out['estimates'].get(v)
            if est is None:
                interp.append(f"{v}: not estimated.")
                continue
            coef = est['coef']
            p = est['pvalue']
            OR = est['OR']
            ci_low = est['CI_2.5%']
            ci_high = est['CI_97.5%']
            # Interpret direction and significance
            sig = (p is not None and p < 0.05)
            direction = "increases" if coef > 0 else ("decreases" if coef < 0 else "no effect")
            interp_line = (f"{v}: coef={coef:.3f}, OR={OR:.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f}), "
                           f"p={p:.3f} -> {direction} focal group's odds of winning" +
                           (", statistically significant" if sig else ", not statistically significant") + ".")
            interp.append(interp_line)
        description = " ".join(interp)

    return {
        "object": out,
        "description": description
    }