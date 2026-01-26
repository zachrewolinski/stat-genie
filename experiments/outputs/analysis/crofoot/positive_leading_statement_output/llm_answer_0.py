def extract_final_answer(model_output):
    """
    Extract key statistics from the model_output produced by the provided modeling function.
    Returns a dict with keys:
      - "object": dict containing coefficients (coef, se, p, 95% CI) for the primary terms,
                  marginal effects if available, and predicted probabilities for illustrative scenarios.
      - "description": a brief, programmatic interpretation of those numbers in plain language.
    """
    import numpy as np
    import pandas as pd

    # Validate model_output structure
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict with keys 'model_result' and optionally 'marginal_effects'.")

    res = model_output.get('model_result', None)
    margeff = model_output.get('marginal_effects', None)

    if res is None:
        raise ValueError("model_output does not contain 'model_result'.")

    out = {}
    coef_table = {}

    # Terms of primary interest
    terms = ['RelGroupSize_z', 'HomeFieldAdv_z', 'RelGroupSize_z:HomeFieldAdv_z']

    # Helper to safely extract param info
    try:
        params = res.params
    except Exception:
        params = None
    try:
        bse = res.bse
    except Exception:
        bse = None
    try:
        pvalues = res.pvalues
    except Exception:
        pvalues = None
    try:
        conf = res.conf_int()
        # conf is DataFrame-like with numeric index matching params
    except Exception:
        conf = None

    for t in terms:
        if params is not None and t in params.index:
            coef = float(params.loc[t])
            se = float(bse.loc[t]) if (bse is not None and t in bse.index) else None
            p = float(pvalues.loc[t]) if (pvalues is not None and t in pvalues.index) else None
            if conf is not None:
                try:
                    ci_low = float(conf.loc[t, 0])
                    ci_high = float(conf.loc[t, 1])
                except Exception:
                    # conf may have column names; try positional
                    try:
                        ci_vals = conf.loc[t].values
                        ci_low, ci_high = float(ci_vals[0]), float(ci_vals[1])
                    except Exception:
                        ci_low, ci_high = None, None
            else:
                ci_low, ci_high = None, None

            coef_table[t] = {
                'coef': coef,
                'se': se,
                'p_value': p,
                '95%_ci': [ci_low, ci_high]
            }
        else:
            coef_table[t] = None

    out['coefficients'] = coef_table

    # Try to extract average marginal effects from margeff if available
    marg_table = {}
    if margeff is not None:
        try:
            # summary_frame() returns a DataFrame with 'dy/dx' etc.
            sf = margeff.summary_frame()
            # Look for rows RelGroupSize_z and HomeFieldAdv_z
            for var in ['RelGroupSize_z', 'HomeFieldAdv_z']:
                if var in sf.index:
                    row = sf.loc[var]
                    marg_table[var] = {
                        'AME': float(row['dy/dx']),
                        'se': float(row['Std. Err.']) if 'Std. Err.' in row.index else float(row['std err']) if 'std err' in row.index else None,
                        'z': float(row['z']) if 'z' in row.index else None,
                        'p_value': float(row['P>|z|']) if 'P>|z|' in row.index else (float(row['p']) if 'p' in row.index else None),
                        '95%_ci': [float(row.get('[0.025', np.nan)) if '[0.025' in row.index else None,
                                   float(row.get('0.975]', np.nan)) if '0.975]' in row.index else None]
                    }
        except Exception:
            marg_table = {}

    out['marginal_effects_from_margeff'] = marg_table if marg_table else None

    # If margeff not available or to aid interpretation, compute predicted probabilities
    # for illustrative combinations of RelGroupSize_z (-1, 0, +1) and HomeFieldAdv_z (-1, 0, +1)
    # holding controls at 0 (mean standardized).
    pred_table = {}
    try:
        # Build grid
        rel_vals = [-1.0, 0.0, 1.0]
        home_vals = [-1.0, 0.0, 1.0]
        rows = []
        for r in rel_vals:
            for h in home_vals:
                rows.append({
                    'RelGroupSize_z': r,
                    'HomeFieldAdv_z': h,
                    'RelMaleCount_z': 0.0,
                    'TotalSize_z': 0.0
                })
        exog = pd.DataFrame(rows)
        # Predict returns probability for GLM family=Binomial
        probs = res.predict(exog)
        # attach to table
        idx = 0
        for r in rel_vals:
            for h in home_vals:
                pred_table[f"Rel_{r}_Home_{h}"] = {
                    'RelGroupSize_z': r,
                    'HomeFieldAdv_z': h,
                    'predicted_prob_win': float(probs.iloc[idx])
                }
                idx += 1
    except Exception:
        pred_table = None

    out['predicted_probabilities_examples'] = pred_table

    # Build a short programmatic description based on extracted stats
    desc_lines = []
    # Helper to format significance
    def signif(p):
        try:
            if p is None:
                return "p=NA"
            if p < 0.001:
                return "p<0.001"
            return f"p={p:.3f}"
        except Exception:
            return "p=NA"

    # Interpret RelGroupSize_z
    rg = coef_table.get('RelGroupSize_z')
    if rg:
        s = f"Relative group size: coef={rg['coef']:.3f}, se={rg['se']:.3f} ({signif(rg['p_value'])}), 95% CI=[{rg['95%_ci'][0]:.3f}, {rg['95%_ci'][1]:.3f}]."
        # Direction
        if rg['p_value'] is not None and rg['p_value'] < 0.05:
            if rg['coef'] > 0:
                s += " Larger focal groups have a higher probability of winning (statistically significant)."
            else:
                s += " Larger focal groups have a lower probability of winning (statistically significant)."
        else:
            if rg['coef'] > 0:
                s += " Point estimate suggests larger focal groups more likely to win, but not statistically significant."
            else:
                s += " Point estimate suggests larger focal groups less likely to win, but not statistically significant."
        desc_lines.append(s)
    else:
        desc_lines.append("Relative group size result not available in model output.")

    # Interpret HomeFieldAdv_z
    hf = coef_table.get('HomeFieldAdv_z')
    if hf:
        s = f"Home-field advantage: coef={hf['coef']:.3f}, se={hf['se']:.3f} ({signif(hf['p_value'])}), 95% CI=[{hf['95%_ci'][0]:.3f}, {hf['95%_ci'][1]:.3f}]."
        if hf['p_value'] is not None and hf['p_value'] < 0.05:
            if hf['coef'] > 0:
                s += " Being closer to home is associated with higher probability of winning (statistically significant)."
            else:
                s += " Being closer to home is associated with lower probability of winning (statistically significant)."
        else:
            s += " Effect not statistically significant."
        desc_lines.append(s)
    else:
        desc_lines.append("Home-field advantage result not available in model output.")

    # Interpret interaction
    inter = coef_table.get('RelGroupSize_z:HomeFieldAdv_z')
    if inter:
        s = f"Interaction (RelGroupSize_z × HomeFieldAdv_z): coef={inter['coef']:.3f}, se={inter['se']:.3f} ({signif(inter['p_value'])}), 95% CI=[{inter['95%_ci'][0]:.3f}, {inter['95%_ci'][1]:.3f}]."
        if inter['p_value'] is not None and inter['p_value'] < 0.05:
            if inter['coef'] > 0:
                s += " Positive interaction: the advantage of being numerically larger increases when the focal group is closer to its home center."
            else:
                s += " Negative interaction: the advantage of being numerically larger decreases when the focal group is closer to its home center."
        else:
            s += " Interaction not statistically significant; no strong evidence that home-field modifies the size advantage."
        desc_lines.append(s)
    else:
        desc_lines.append("Interaction result not available in model output.")

    # Add marginal effects note if available
    if out['marginal_effects_from_margeff']:
        me = out['marginal_effects_from_margeff']
        for var, info in me.items():
            desc_lines.append(f"Average marginal effect of {var}: AME={info['AME']:.4f}, se={info.get('se')}, p={info.get('p_value')}.")
    else:
        desc_lines.append("Average marginal effects from statsmodels get_margeff not available.")

    # Add a short summary of predicted probabilities (examples)
    if pred_table:
        # Show effect of moving RelGroupSize_z from -1 to +1 at HomeFieldAdv_z = -1,0,1
        for h in [-1.0, 0.0, 1.0]:
            p_minus = pred_table[f"Rel_-1.0_Home_{h}"]['predicted_prob_win']
            p_plus = pred_table[f"Rel_1.0_Home_{h}"]['predicted_prob_win']
            diff = p_plus - p_minus
            desc_lines.append(f"Illustration: when HomeFieldAdv_z={h}, predicted P(win) goes from {p_minus:.3f} (RelGroupSize_z=-1) to {p_plus:.3f} (RelGroupSize_z=+1), difference ≈ {diff:.3f}. Controls held at mean (0).")

    description = " ".join(desc_lines)

    return {
        "object": out,
        "description": description
    }