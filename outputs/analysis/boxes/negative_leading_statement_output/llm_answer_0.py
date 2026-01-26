def extract_final_answer(model_output):
    """
    Extracts age-related effects (main and culture-moderated) from the fitted models
    contained in `model_output` (the dict returned by the modeling function).
    Returns a dict with keys:
      - "object": nested dict with numeric results (coefficients, SEs, z, p, 95% CI)
      - "description": plain-English summary of what the numbers mean
    
    This function is defensive: it handles exceptions stored in model_output,
    and falls back to approximate CIs when needed.
    """
    import numpy as np
    from math import sqrt
    from scipy import stats

    out = {
        "multinomial_age_results": None,
        "social_reliance_age_result": None,
        "majority_pref_logit_error": None
    }

    descriptions = []

    # Helper to format CI if conf_int returned in different shapes
    def get_ci(res, outcome, param, coef, se):
        try:
            ci = res.conf_int()
            # conf_int for MNLogit returns a DataFrame whose index matches params' MultiIndex
            # Try several access patterns:
            try:
                lower, upper = ci.loc[(outcome, param)].values
            except Exception:
                try:
                    # maybe index is strings like "outcome[param]"
                    lower, upper = ci.loc[outcome, param]
                except Exception:
                    # fallback to normal approximation
                    z = stats.norm.ppf(0.975)
                    lower, upper = coef - z * se, coef + z * se
        except Exception:
            z = stats.norm.ppf(0.975)
            lower, upper = coef - z * se, coef + z * se
        return float(lower), float(upper)

    # ---------- Multinomial: age effects ----------
    mn_res = model_output.get('multinomial_result')
    exog_cols = model_output.get('exog_multinomial_columns', [])
    if isinstance(mn_res, Exception):
        out['multinomial_age_results'] = {"error": str(mn_res)}
        descriptions.append("Multinomial model failed; cannot extract age effects.")
    else:
        try:
            res = mn_res  # statsmodels results wrapper
            params = res.params  # DataFrame: index = outcome labels, columns = exog names
            bse = res.bse
            pvals = res.pvalues
            cov = res.cov_params()  # covariance matrix; index and cols are MultiIndex (outcome,param)

            # Identify culture dummy names from exog columns (those that start with 'culture_' and do not contain '_x_age')
            culture_dummies = [c for c in exog_cols if c.startswith('culture_') and not c.endswith('_x_age')]
            # Identify interaction names (age x culture)
            interaction_names = [c for c in exog_cols if c.endswith('_x_age') and c.startswith('culture_')]

            # Determine the omitted (reference) culture label: unknown name, so label as 'reference (omitted)'
            # Number of cultures = len(culture_dummies) + 1
            cultures = ['reference_omitted'] + culture_dummies

            # For each non-reference outcome (rows of params), extract age coefficient and cultural moderation
            multi_results = {}
            for outcome in params.index:
                row = params.loc[outcome]
                se_row = bse.loc[outcome]
                p_row = pvals.loc[outcome]

                # Main age effect (this is the coeff on age_c for this outcome vs reference outcome)
                if 'age_c' in row.index:
                    age_coef = float(row['age_c'])
                    age_se = float(se_row['age_c'])
                    age_p = float(p_row['age_c'])
                    age_ci = get_ci(res, outcome, 'age_c', age_coef, age_se)
                else:
                    age_coef = None
                    age_se = None
                    age_p = None
                    age_ci = (None, None)

                # Build per-culture combined age effects: baseline = age_coef; for other cultures add interaction term
                per_culture = {}
                # baseline culture
                per_culture['reference_omitted'] = {
                    'coef': age_coef,
                    'se': age_se,
                    'p': age_p,
                    '95ci': age_ci,
                    'note': 'Effect of age in the omitted/reference culture (no culture dummy present).'
                }

                # For each observed culture dummy, compute combined effect = age_coef + interaction_coef
                for cd in culture_dummies:
                    inter_name = f'{cd}_x_age'
                    inter_coef = float(row[inter_name]) if inter_name in row.index else 0.0
                    # Retrieve covariance elements if possible to compute SE of sum
                    try:
                        var_age = float(cov.loc[(outcome, 'age_c'), (outcome, 'age_c')])
                        var_inter = float(cov.loc[(outcome, inter_name), (outcome, inter_name)]) if inter_name in row.index else 0.0
                        covar = float(cov.loc[(outcome, 'age_c'), (outcome, inter_name)]) if inter_name in row.index else 0.0
                        combined_var = var_age + var_inter + 2.0 * covar
                        combined_se = sqrt(max(combined_var, 0.0))
                    except Exception:
                        # fallback: naive combination ignoring covariance (conservative if covariance unknown)
                        try:
                            se_age = float(se_row['age_c'])
                            se_inter = float(se_row[inter_name]) if inter_name in se_row.index else 0.0
                            combined_se = sqrt(se_age ** 2 + se_inter ** 2)
                        except Exception:
                            combined_se = None

                    combined_coef = (age_coef if age_coef is not None else 0.0) + inter_coef
                    if combined_se is not None and combined_se > 0:
                        z = combined_coef / combined_se
                        p_comb = 2.0 * stats.norm.sf(abs(z))
                        # Compute CI
                        z975 = stats.norm.ppf(0.975)
                        ci_lower = combined_coef - z975 * combined_se
                        ci_upper = combined_coef + z975 * combined_se
                    else:
                        p_comb = None
                        ci_lower, ci_upper = None, None

                    per_culture[cd] = {
                        'coef': float(combined_coef),
                        'se': float(combined_se) if combined_se is not None else None,
                        'p': float(p_comb) if p_comb is not None else None,
                        '95ci': (float(ci_lower) if ci_lower is not None else None,
                                 float(ci_upper) if ci_upper is not None else None),
                        'note': f'Effect of age for children in culture represented by dummy "{cd}" '
                                '(this is age main effect + culture_x_age interaction).'
                    }

                multi_results[str(outcome)] = {
                    'age_main_coef': float(age_coef) if age_coef is not None else None,
                    'age_main_se': float(age_se) if age_se is not None else None,
                    'age_main_p': float(age_p) if age_p is not None else None,
                    'age_main_95ci': age_ci,
                    'per_culture_age_effects': per_culture
                }

            out['multinomial_age_results'] = {
                'outcome_vs_reference': multi_results,
                'info': {
                    'note': 'Each outcome row is a comparison of that choice against the multinomial reference outcome used by the model. '
                            'Per-culture age effects are calculated as (age main effect) + (culture_x_age interaction) when present. '
                            'The omitted (reference) culture is the category for which no culture dummy was created (labelled "reference_omitted").',
                    'culture_dummies_in_model': culture_dummies,
                    'interaction_terms_in_model': interaction_names
                }
            }
            descriptions.append("Extracted age coefficients and culture-moderated age effects from the multinomial model.")
        except Exception as e:
            out['multinomial_age_results'] = {"error": f"Failed to extract results: {e}"}
            descriptions.append(f"Failed to extract multinomial details: {e}")

    # ---------- SocialReliance logistic: age effect ----------
    logit_res = model_output.get('social_reliance_logit_result')
    if isinstance(logit_res, Exception):
        out['social_reliance_age_result'] = {"error": str(logit_res)}
        descriptions.append("SocialReliance logistic model failed; cannot extract age effect.")
    else:
        try:
            res2 = logit_res
            params2 = res2.params
            bse2 = res2.bse
            p2 = res2.pvalues
            if 'age_c' in params2.index:
                coef = float(params2['age_c'])
                se = float(bse2['age_c'])
                z = coef / se if se != 0 else None
                pval = float(p2['age_c'])
                z975 = stats.norm.ppf(0.975)
                ci = (coef - z975 * se, coef + z975 * se)
                # convert to odds ratio
                orr = float(np.exp(coef))
                or_ci = (float(np.exp(ci[0])), float(np.exp(ci[1])))
                out['social_reliance_age_result'] = {
                    'coef_age_c': coef,
                    'se': se,
                    'z': float(z) if z is not None else None,
                    'p': pval,
                    '95ci_coef': (float(ci[0]), float(ci[1])),
                    'odds_ratio': orr,
                    '95ci_odds_ratio': (or_ci[0], or_ci[1]),
                    'note': 'Effect of centered age on log-odds of relying on social information (SocialReliance==1).'
                }
                descriptions.append("Extracted age effect from the SocialReliance logistic model (log-odds and OR).")
            else:
                out['social_reliance_age_result'] = {"error": "age_c not found in logistic model parameters"}
                descriptions.append("age_c not present in SocialReliance model parameters.")
        except Exception as e:
            out['social_reliance_age_result'] = {"error": f"Failed to extract: {e}"}
            descriptions.append(f"Failed to extract SocialReliance logistic details: {e}")

    # ---------- Majority-pref logistic error (if any) ----------
    maj_pref = model_output.get('majority_pref_logit_result')
    if isinstance(maj_pref, Exception):
        out['majority_pref_logit_error'] = str(maj_pref)
        descriptions.append("Majority-preference logistic model was not available (error recorded).")
    else:
        # if model exists, we could extract similarly; but modeling code indicated this often failed
        try:
            res3 = maj_pref
            if hasattr(res3, 'params') and 'age_c' in res3.params.index:
                coef = float(res3.params['age_c'])
                se = float(res3.bse['age_c'])
                z = coef / se if se != 0 else None
                pval = float(res3.pvalues['age_c'])
                z975 = stats.norm.ppf(0.975)
                ci = (coef - z975 * se, coef + z975 * se)
                out['majority_pref_logit_result'] = {
                    'coef_age_c': coef,
                    'se': se,
                    'z': float(z) if z is not None else None,
                    'p': pval,
                    '95ci_coef': (float(ci[0]), float(ci[1])),
                    'note': 'Effect of age on probability of choosing the majority option among social users.'
                }
                descriptions.append("Extracted age effect from majority-preference logistic model.")
            else:
                out['majority_pref_logit_result'] = {"info": "No majority-preference logistic result to extract or age_c missing."}
        except Exception as e:
            out['majority_pref_logit_result'] = {"error": f"Failed to extract majority-pref model details: {e}"}
            descriptions.append(f"Failed to extract majority-pref logistic details: {e}")

    # Build a compact human-readable description
    human_description_lines = []
    human_description_lines.append("What was extracted:")
    human_description_lines.append("- Multinomial model: coefficients for 'age_c' (main) and culture-specific age effects "
                                   "(computed as age + culture_x_age where available) for each non-reference outcome.")
    human_description_lines.append("- SocialReliance logistic: age effect (log-odds, SE, p, 95% CI, and odds ratio).")
    if out['majority_pref_logit_error']:
        human_description_lines.append(f"- Majority-pref logistic model: not available / errored: {out['majority_pref_logit_error']}")
    human_description_lines.append("")
    human_description_lines.append("Notes on interpretation:")
    human_description_lines.append("- Multinomial coefficients are log-odds changes in choosing that outcome versus the model's reference outcome for a 1-unit increase in centered age.")
    human_description_lines.append("- Per-culture age effects are computed relative to the omitted (reference) culture; the omitted culture is labelled 'reference_omitted' in the returned object because the original category name is not present in the model matrix.")
    human_description_lines.append("- Statistical significance is reported via p-values; 95% CIs and SEs are provided. Where exact covariance information is available, combined SEs for sums (age + interaction) were computed using the covariance matrix; otherwise a conservative approximation was used.")

    description_text = "\n".join(human_description_lines + ["", "Summary of extraction actions:"] + descriptions)

    return {"object": out, "description": description_text}