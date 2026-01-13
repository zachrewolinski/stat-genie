def extract_final_answer(model_output):
    """
    Extracts the estimated effect of Reader View for readers with dyslexia from a fitted
    statsmodels MixedLMResults (or wrapper) object that used the formula:
      log_speed ~ reader_view * dyslexia_bin + ...
    Returns a dictionary with:
      - "object": a dict of numeric results (coef on log scale, se, z, p, CI, percent change)
      - "description": a short plain-English interpretation of the result

    The function is written defensively to find parameter names like:
      'reader_view' (main effect) and an interaction parameter that contains both
      'reader_view' and 'dyslexia' (e.g., 'reader_view:dyslexia_bin').
    """
    import numpy as np
    from math import sqrt
    from scipy import stats

    res = model_output

    # Try to obtain parameter estimates and covariance matrix from common attributes
    # statsmodels MixedLMResultsWrapper exposes .params and .cov_params()
    try:
        params = res.params
    except Exception:
        # fallback to attribute name used by some statsmodels versions
        try:
            params = res.fe_params
        except Exception as e:
            raise ValueError("Could not extract parameter estimates from model_output") from e

    try:
        cov = res.cov_params()
    except Exception:
        # fallback to covariance of fixed effects
        try:
            cov = res.cov_re  # not ideal, but try something sensible
        except Exception as e:
            raise ValueError("Could not extract covariance matrix from model_output") from e

    # Ensure params is a pandas Series-like with index
    try:
        param_names = list(params.index)
    except Exception:
        # if params is a numpy array, we cannot proceed reliably
        raise ValueError("Parameter object from model_output does not have named indices")

    # Find parameter names related to reader_view
    reader_terms = [n for n in param_names if 'reader_view' in n]
    if not reader_terms:
        raise ValueError("No parameter name containing 'reader_view' was found in the model parameters.")

    # Identify interaction term (contains both reader_view and dyslexia)
    interaction_term = None
    for n in reader_terms:
        if ('dyslex' in n) or ('dyslexia' in n) or ('dyslexia_bin' in n):
            interaction_term = n
            break

    # Identify main reader_view term (a reader_view term that is not the interaction)
    main_term = None
    for n in reader_terms:
        if n != interaction_term:
            main_term = n
            break

    if main_term is None:
        # If we could not find a clear main term but there is exactly one reader_terms entry,
        # treat that as the main effect and assume no separate interaction term.
        if len(reader_terms) == 1 and interaction_term is None:
            main_term = reader_terms[0]
        else:
            raise ValueError("Could not identify the main 'reader_view' coefficient in parameters.")

    # Extract coefficients
    beta_main = float(params[main_term])
    # covariance entries
    try:
        var_main = float(cov.loc[main_term, main_term])
    except Exception:
        # try indexing if cov is a numpy matrix with same ordering as params
        try:
            idx_main = param_names.index(main_term)
            var_main = float(cov[idx_main, idx_main])
        except Exception as e:
            raise ValueError("Could not find variance for main term in covariance matrix") from e

    if interaction_term is not None:
        beta_inter = float(params[interaction_term])
        try:
            var_inter = float(cov.loc[interaction_term, interaction_term])
            cov_mi = float(cov.loc[main_term, interaction_term])
        except Exception:
            # fallback to index-based access
            try:
                idx_main = param_names.index(main_term)
                idx_inter = param_names.index(interaction_term)
                var_inter = float(cov[idx_inter, idx_inter])
                cov_mi = float(cov[idx_main, idx_inter])
            except Exception as e:
                raise ValueError("Could not find covariance entries for interaction/main terms") from e

        # Combined effect of reader_view for dyslexia = main + interaction
        beta_combined = beta_main + beta_inter
        var_combined = var_main + var_inter + 2.0 * cov_mi
        se_combined = sqrt(var_combined) if var_combined >= 0 else float('nan')

        # z-stat and p-value using normal approximation
        z_combined = beta_combined / se_combined if se_combined != 0 else float('nan')
        p_combined = 2.0 * (1.0 - stats.norm.cdf(abs(z_combined))) if se_combined != 0 else float('nan')

        # 95% CI on log scale
        ci_lo = beta_combined - 1.96 * se_combined
        ci_hi = beta_combined + 1.96 * se_combined

        # Convert to percent change in original reading speed
        pct_change = (np.exp(beta_combined) - 1.0) * 100.0
        pct_ci_lo = (np.exp(ci_lo) - 1.0) * 100.0
        pct_ci_hi = (np.exp(ci_hi) - 1.0) * 100.0

        result_obj = {
            "term_for_dyslexic_log_coef": float(beta_combined),
            "term_for_dyslexic_se": float(se_combined),
            "term_for_dyslexic_z": float(z_combined),
            "term_for_dyslexic_pvalue": float(p_combined),
            "term_for_dyslexic_95ci_log": (float(ci_lo), float(ci_hi)),
            "term_for_dyslexic_percent_change": float(pct_change),
            "term_for_dyslexic_95ci_percent": (float(pct_ci_lo), float(pct_ci_hi)),
            "main_term_name": main_term,
            "interaction_term_name": interaction_term,
            "main_term_log_coef": float(beta_main),
            "interaction_log_coef": float(beta_inter)
        }

        # Build a concise description
        sig_text = "statistically significant (p < 0.05)" if p_combined < 0.05 else "not statistically significant (p >= 0.05)"
        direction = "increase" if beta_combined > 0 else ("decrease" if beta_combined < 0 else "no change")
        description = (
            f"The estimated effect of turning Reader View ON for readers with dyslexia is a log-coefficient of "
            f"{beta_combined:.4f} (SE={se_combined:.4f}, z={z_combined:.2f}, p={p_combined:.3g}). "
            f"On the original reading-speed scale this corresponds to a {pct_change:.1f}% {direction} "
            f"in reading speed (95% CI: {pct_ci_lo:.1f}% to {pct_ci_hi:.1f}%). The effect is {sig_text}. "
            f"(Main reader_view term = {beta_main:.4f}; interaction term = {beta_inter:.4f}.)"
        )

    else:
        # No interaction term found: effect for dyslexic readers equals main effect
        se_main = sqrt(var_main) if var_main >= 0 else float('nan')
        z_main = beta_main / se_main if se_main != 0 else float('nan')
        p_main = 2.0 * (1.0 - stats.norm.cdf(abs(z_main))) if se_main != 0 else float('nan')
        ci_lo = beta_main - 1.96 * se_main
        ci_hi = beta_main + 1.96 * se_main
        pct_change = (np.exp(beta_main) - 1.0) * 100.0
        pct_ci_lo = (np.exp(ci_lo) - 1.0) * 100.0
        pct_ci_hi = (np.exp(ci_hi) - 1.0) * 100.0

        result_obj = {
            "term_for_dyslexic_log_coef": float(beta_main),
            "term_for_dyslexic_se": float(se_main),
            "term_for_dyslexic_z": float(z_main),
            "term_for_dyslexic_pvalue": float(p_main),
            "term_for_dyslexic_95ci_log": (float(ci_lo), float(ci_hi)),
            "term_for_dyslexic_percent_change": float(pct_change),
            "term_for_dyslexic_95ci_percent": (float(pct_ci_lo), float(pct_ci_hi)),
            "main_term_name": main_term,
            "interaction_term_name": None,
            "main_term_log_coef": float(beta_main)
        }

        sig_text = "statistically significant (p < 0.05)" if p_main < 0.05 else "not statistically significant (p >= 0.05)"
        direction = "increase" if beta_main > 0 else ("decrease" if beta_main < 0 else "no change")
        description = (
            f"No reader_view × dyslexia interaction parameter was found. The estimated effect of turning Reader View ON "
            f"(applies equally to dyslexic readers under this model) is a log-coefficient of {beta_main:.4f} "
            f"(SE={se_main:.4f}, z={z_main:.2f}, p={p_main:.3g}). On the original reading-speed scale this corresponds "
            f"to a {pct_change:.1f}% {direction} in reading speed (95% CI: {pct_ci_lo:.1f}% to {pct_ci_hi:.1f}%). "
            f"The effect is {sig_text}."
        )

    return {"object": result_obj, "description": description}