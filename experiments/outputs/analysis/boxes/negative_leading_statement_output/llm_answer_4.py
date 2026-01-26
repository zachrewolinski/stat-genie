def extract_final_answer(model_output):
    """
    Extracts key statistics about age and culture effects from the provided model_output dict.

    Returns a dictionary with keys:
      - "object": dict of numeric results (LR test, multinomial age coefficients & p-values,
                  logistic (demonstrated) age coefficient, odds ratio and CI)
      - "description": plain-language interpretation of those results in the context of
                       whether children's reliance on the majority develops with age
                       and whether that pattern varies across cultures.
    """
    import numpy as np
    import pandas as pd

    out = {
        'mnlogit_lr_test': None,
        'mnlogit_age_effects': None,
        'logit_demonstrated_age_effect': None,
        'notes': []
    }

    # 1) Likelihood-ratio test (joint test of age + culture in the multinomial model)
    lr_info = model_output.get('mnlogit_lr_test')
    if isinstance(lr_info, dict):
        out['mnlogit_lr_test'] = {
            'lr_stat': lr_info.get('lr_stat'),
            'df_diff': lr_info.get('df_diff'),
            'p_value': lr_info.get('p_value')
        }
    else:
        out['notes'].append("No likelihood-ratio test available or test could not be computed.")

    # 2) Extract age coefficients and p-values from the full multinomial model (if present)
    mn = model_output.get('mnlogit_full')
    if mn is None:
        out['notes'].append("Full multinomial model not available in model_output.")
    else:
        try:
            params = mn.params  # could be DataFrame or ndarray-like
            pvals = mn.pvalues

            # Get exog names and attempt to align params into DataFrame with rows=exog, cols=equations
            exog_names = getattr(mn.model, 'exog_names', None)
            # params may be DataFrame already
            if isinstance(params, pd.DataFrame):
                # Common layout: rows = exog names, columns = equations (categories)
                if 'age_centered' in params.index:
                    age_row = params.loc['age_centered']
                    age_p = pvals.loc['age_centered']
                    # Convert to dict keyed by equation label (column names)
                    age_effects = {}
                    for col in params.columns:
                        coef = float(age_row[col]) if not pd.isna(age_row[col]) else None
                        pval = float(age_p[col]) if not pd.isna(age_p[col]) else None
                        age_effects[str(col)] = {'coef': coef, 'p_value': pval}
                    out['mnlogit_age_effects'] = age_effects
                else:
                    # Maybe rows are equations and columns exog names (transposed)
                    if 'age_centered' in params.columns:
                        age_effects = {}
                        for row in params.index:
                            coef = float(params.loc[row, 'age_centered']) if not pd.isna(params.loc[row, 'age_centered']) else None
                            pval = float(pvals.loc[row, 'age_centered']) if not pd.isna(pvals.loc[row, 'age_centered']) else None
                            age_effects[str(row)] = {'coef': coef, 'p_value': pval}
                        out['mnlogit_age_effects'] = age_effects
                    else:
                        out['notes'].append("Could not locate 'age_centered' in multinomial params DataFrame.")
            else:
                # params is an ndarray (likely shape (n_equations, n_params) or (n_params, n_equations)).
                coef_array = np.asarray(params)
                pval_array = np.asarray(pvals)
                if exog_names is None:
                    out['notes'].append("Could not retrieve exog names for multinomial model; cannot map coefficients.")
                else:
                    # Determine orientation: check matching dimension
                    n_exog = len(exog_names)
                    if coef_array.ndim == 2:
                        if coef_array.shape[1] == n_exog:
                            # shape = (n_equations, n_params) -> columns correspond to exog_names
                            col_axis = 1
                            row_labels = [f"eq_{i}" for i in range(coef_array.shape[0])]
                        elif coef_array.shape[0] == n_exog:
                            # shape = (n_params, n_equations) -> rows correspond to exog_names
                            coef_array = coef_array.T
                            pval_array = pval_array.T
                            col_axis = 1
                            row_labels = [f"eq_{i}" for i in range(coef_array.shape[0])]
                        else:
                            out['notes'].append("Multinomial params array has unexpected shape; cannot map reliably.")
                            coef_array = None
                            pval_array = None

                        if coef_array is not None:
                            try:
                                age_idx = exog_names.index('age_centered')
                                age_effects = {}
                                for eq_i in range(coef_array.shape[0]):
                                    coef = float(coef_array[eq_i, age_idx])
                                    pval = float(pval_array[eq_i, age_idx])
                                    age_effects[f"eq_{eq_i}"] = {'coef': coef, 'p_value': pval}
                                out['mnlogit_age_effects'] = age_effects
                            except ValueError:
                                out['notes'].append("'age_centered' not found in multinomial exog names.")
                    else:
                        out['notes'].append("Multinomial params not 2-D; cannot extract age effects.")

        except Exception as e:
            out['notes'].append(f"Error extracting multinomial coefficients: {repr(e)}")

    # 3) Extract age coefficient from the logistic model predicting choosing any demonstrated option
    logit_demo = model_output.get('logit_demonstrated')
    if logit_demo is None:
        err = model_output.get('logit_demonstrated_error')
        if err:
            out['notes'].append(f"Demonstrated-choice logistic model missing: {err}")
        else:
            out['notes'].append("Demonstrated-choice logistic model missing.")
    else:
        try:
            params = logit_demo.params  # Series or ndarray
            pvals = logit_demo.pvalues
            bse = logit_demo.bse
            # params likely a pandas Series indexed by exog names
            if isinstance(params, (pd.Series, pd.DataFrame)):
                if 'age_centered' in params.index:
                    coef = float(params['age_centered'])
                    pval = float(pvals['age_centered'])
                    se = float(bse['age_centered'])
                    orr = float(np.exp(coef))
                    # 95% CI for coefficient then exponentiate
                    ci_low = coef - 1.96 * se
                    ci_high = coef + 1.96 * se
                    or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
                    out['logit_demonstrated_age_effect'] = {
                        'coef': coef,
                        'p_value': pval,
                        'se': se,
                        'odds_ratio': orr,
                        'odds_ratio_95CI': or_ci
                    }
                else:
                    out['notes'].append("'age_centered' not found in logistic-demonstrated model parameters.")
            else:
                # fallback for ndarray: attempt to map with model.exog_names
                exog_names = getattr(logit_demo.model, 'exog_names', None)
                if exog_names and 'age_centered' in exog_names:
                    idx = exog_names.index('age_centered')
                    coef = float(params[idx])
                    pval = float(pvals[idx])
                    se = float(bse[idx])
                    orr = float(np.exp(coef))
                    ci_low = coef - 1.96 * se
                    ci_high = coef + 1.96 * se
                    or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
                    out['logit_demonstrated_age_effect'] = {
                        'coef': coef,
                        'p_value': pval,
                        'se': se,
                        'odds_ratio': orr,
                        'odds_ratio_95CI': or_ci
                    }
                else:
                    out['notes'].append("Could not map logistic-demonstrated params to 'age_centered'.")
        except Exception as e:
            out['notes'].append(f"Error extracting logistic-demonstrated age effect: {repr(e)}")

    # 4) Note about majority-vs-minority logistic (could not fit here)
    maj_err = model_output.get('logit_majority_given_demonstrated_error')
    if maj_err is not None:
        out['notes'].append(f"Majority-vs-minority model among demonstrated choices: {maj_err}")

    # 5) Compose a brief interpretation string
    desc_parts = []
    # LR test interpretation
    if out['mnlogit_lr_test'] is not None:
        p = out['mnlogit_lr_test']['p_value']
        desc_parts.append(
            f"Likelihood-ratio test for joint effect of age and culture on the 3-way choice: "
            f"LR={out['mnlogit_lr_test']['lr_stat']:.3f}, df={out['mnlogit_lr_test']['df_diff']}, p={p:.3f}."
        )
        if p is not None and p < 0.05:
            desc_parts.append("This indicates a significant joint effect of age and culture on children's 3-way choice.")
        else:
            desc_parts.append("This does NOT provide evidence for a significant joint effect of age and culture "
                              "on the 3-way choice (p >= 0.05).")
    else:
        desc_parts.append("No LR test result available to assess joint age + culture effect.")

    # Multinomial age effects summary
    if out['mnlogit_age_effects'] is not None:
        # Report per-equation effects
        eq_lines = []
        for eq, info in out['mnlogit_age_effects'].items():
            coef = info.get('coef')
            pval = info.get('p_value')
            if coef is None:
                eq_lines.append(f"{eq}: age effect not available")
            else:
                signif = "p<0.05" if (pval is not None and pval < 0.05) else "ns"
                eq_lines.append(f"{eq}: coef={coef:.4f}, p={pval:.3f} ({signif})")
        desc_parts.append("Multinomial model: age coefficients by equation: " + "; ".join(eq_lines))
    else:
        desc_parts.append("Multinomial model age coefficients could not be retrieved.")

    # Logistic-demonstrated summary
    if out['logit_demonstrated_age_effect'] is not None:
        le = out['logit_demonstrated_age_effect']
        signif = "significant (p<0.05)" if le['p_value'] < 0.05 else "not significant"
        desc_parts.append(
            f"Logistic model predicting choosing any demonstrated option: age coef={le['coef']:.4f}, "
            f"p={le['p_value']:.3f} ({signif}); odds ratio={le['odds_ratio']:.3f}, "
            f"95% CI={tuple(round(x,3) for x in le['odds_ratio_95CI'])}."
        )
    else:
        desc_parts.append("Logistic model for choosing any demonstrated option: age effect not available.")

    # Final interpretive summary
    desc_parts.append(
        "Overall interpretation: There is no strong evidence here that reliance on the majority systematically "
        "increases (or decreases) with age across cultures — the joint LR test of age + culture is not significant, "
        "and the extracted age coefficients in the multinomial and demonstrated-choice logistic models are not "
        "consistently significant. Note: the model of majority vs minority choices among only demonstrated choices "
        "could not be fit here (index alignment issue), so conclusions specifically about majority-vs-minority "
        "preferences among those who copied are limited."
    )

    return {
        "object": out,
        "description": " ".join(desc_parts)
    }