def extract_final_answer(model_output):
    """
    Extract statistics related to the effect of instructor beauty from a fitted
    statsmodels RegressionResultsWrapper (model_output).

    Returns a dictionary with keys:
      - "object": a dict containing:
          * 'estimates': coefficient, clustered SE, t-value, p-value, and 95% CI
                         for beauty_z and beauty_z2 (if present)
          * 'marginal_effects': marginal effect of beauty (d eval / d beauty)
                                evaluated at beauty_z = -1, 0, +1 (if both terms present)
          * 'joint_test': result of joint test that both beauty_z and beauty_z2 = 0
                         (dictionary with F statistic and p-value when available)
      - "description": brief interpretation of these statistics in context
    """
    import numpy as np

    res = model_output

    out_estimates = {}
    terms = ['beauty_z', 'beauty_z2']

    # Ensure model_result has required attributes
    params = getattr(res, 'params', None)
    if params is None:
        return {
            "object": None,
            "description": "The provided model_output does not have .params; cannot extract estimates."
        }

    # Extract term-level statistics if present
    conf = None
    try:
        conf = res.conf_int()
    except Exception:
        conf = None

    for t in terms:
        if t in params.index:
            coef = float(params[t])
            se = float(res.bse[t]) if hasattr(res, 'bse') and t in res.bse.index else None
            tval = float(res.tvalues[t]) if hasattr(res, 'tvalues') and t in res.tvalues.index else None
            pval = float(res.pvalues[t]) if hasattr(res, 'pvalues') and t in res.pvalues.index else None
            ci = None
            if conf is not None and t in conf.index:
                ci = [float(conf.loc[t, 0]), float(conf.loc[t, 1])]
            out_estimates[t] = {
                "coef": coef,
                "se_clustered": se,
                "t_value": tval,
                "p_value": pval,
                "ci_95": ci
            }

    # If both linear and quadratic present, compute marginal effects at several beauty levels
    marginal_effects = None
    if ('beauty_z' in out_estimates) and ('beauty_z2' in out_estimates):
        b1 = out_estimates['beauty_z']['coef']
        b2 = out_estimates['beauty_z2']['coef']
        # marginal effect = d(eval)/d(beauty_z) = b1 + 2*b2*beauty_z
        marg_points = [-1.0, 0.0, 1.0]
        marginal_effects = {}
        for z in marg_points:
            marginal_effects[z] = float(b1 + 2.0 * b2 * z)

    # Joint test that both beauty coefficients are zero
    joint_test = None
    if all(t in params.index for t in terms):
        try:
            # Try string formulation first
            ft = res.f_test("beauty_z = 0, beauty_z2 = 0")
            # ContrastResult typically has .fvalue and .pvalue attributes (or .fvalue array)
            fval = None
            pval = None
            if hasattr(ft, 'fvalue'):
                # ft.fvalue can be array-like
                try:
                    fval = float(np.asarray(ft.fvalue).squeeze())
                except Exception:
                    fval = None
            if hasattr(ft, 'pvalue'):
                try:
                    pval = float(np.asarray(ft.pvalue).squeeze())
                except Exception:
                    pval = None
            # Some versions expose .pf or other names; attempt to retrieve if missing
            if pval is None:
                for attr in ('pf', 'pval', 'prob_f'):
                    if hasattr(ft, attr):
                        try:
                            pval = float(getattr(ft, attr))
                            break
                        except Exception:
                            pass
            joint_test = {"F": fval, "p_value": pval}
        except Exception:
            # As a fallback, attempt matrix formulation
            try:
                idx = list(params.index)
                k = len(params)
                R = np.zeros((2, k))
                R[0, idx.index('beauty_z')] = 1.0
                R[1, idx.index('beauty_z2')] = 1.0
                ft = res.f_test(R)
                fval = float(np.asarray(ft.fvalue).squeeze()) if hasattr(ft, 'fvalue') else None
                pval = float(np.asarray(ft.pvalue).squeeze()) if hasattr(ft, 'pvalue') else None
                joint_test = {"F": fval, "p_value": pval}
            except Exception:
                joint_test = {"error": "Joint test failed or not available."}

    result_object = {
        "estimates": out_estimates,
        "marginal_effects_at_beauty_z": marginal_effects,
        "joint_test_beauty_linear_and_quadratic": joint_test
    }

    # Short interpretation / description
    # Note: Interpretation depends on sign/magnitude and p-values:
    #  - Positive marginal effect means more attractive instructors tend to receive higher evals.
    #  - Negative marginal effect means the opposite.
    #  - p-values indicate whether each coefficient (or both jointly) are statistically distinguishable from zero.
    description_lines = [
        "This output provides the estimated coefficients, clustered standard errors, t-values,",
        "p-values, and 95% confidence intervals for beauty_z (linear) and beauty_z2 (quadratic),",
        "if present in the model. Because the model includes a quadratic term, the marginal",
        "effect of beauty on evaluation is b1 + 2*b2*beauty_z; marginal effects are shown at",
        "beauty_z = -1, 0, and +1 (one SD below mean, mean, one SD above mean).",
        "Also included is a joint F-test of whether both beauty_z and beauty_z2 are zero.",
        "Interpretation: positive marginal effect => higher attractiveness associated with",
        "higher evaluation scores; significance determined by the provided p-values."
    ]
    description = " ".join(description_lines)

    return {"object": result_object, "description": description}