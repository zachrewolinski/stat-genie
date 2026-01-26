def extract_final_answer(model_output):
    """
    Extract key statistics from the fitted models returned by the provided modeling function.

    Returns a dictionary with keys:
      - "object": a serializable dict containing numeric results (coefficients, SEs, p-values,
                  CIs) for each parameter, a summary of Age-related terms and Age x Culture
                  interaction tests, and predicted probability change across the observed
                  age range for each culture (holding other covariates at sample means).
      - "description": a short plain-language explanation of how to interpret the returned
                       numbers in relation to the research question:
                       "How do children's reliance on majority preference develop with age
                        across different cultural contexts?"
    """
    import numpy as np
    import pandas as pd

    out = {"object": {}, "description": None}

    # Helper to summarize a statsmodels binary results object
    def summarize_result(res):
        summary = {}
        # Parameter table
        params = res.params
        bse = res.bse
        pvals = res.pvalues
        try:
            ci = res.conf_int()
            ci.columns = ['ci_lower', 'ci_upper']
        except Exception:
            # If conf_int fails for some reason, fill with NaNs
            ci = pd.DataFrame(index=params.index, data={'ci_lower': np.nan, 'ci_upper': np.nan})

        df_params = pd.DataFrame({
            'coef': params,
            'se': bse,
            'pval': pvals,
            'ci_lower': ci['ci_lower'],
            'ci_upper': ci['ci_upper']
        })

        # Convert param table to nested dict for serialization
        params_dict = df_params.to_dict(orient='index')
        summary['params'] = params_dict

        # Identify Age terms and Age x Culture interaction terms
        age_terms = {name: params_dict[name] for name in params_dict.keys() if name in ('Age_c', 'Age2')}
        interaction_terms = {name: params_dict[name] for name in params_dict.keys() if 'Age_c:C(culture)' in name}

        summary['age_terms'] = age_terms
        summary['interaction_terms'] = interaction_terms

        # Flag whether any interaction term is individually statistically significant (p < .05)
        any_interaction_sig = any((term_info['pval'] < 0.05) for term_info in interaction_terms.values()) if interaction_terms else False
        summary['any_interaction_term_significant'] = bool(any_interaction_sig)

        # Joint Wald test for all Age_c:C(culture) coefficients = 0 (if there are any such terms)
        joint_pval = None
        if interaction_terms:
            try:
                param_names = list(params.index)
                # indices of interaction parameters
                idxs = [param_names.index(name) for name in interaction_terms.keys()]
                # Build R matrix: one row per tested parameter selecting that parameter
                R = np.zeros((len(idxs), len(param_names)))
                for i, idx in enumerate(idxs):
                    R[i, idx] = 1.0
                wres = res.wald_test(R)
                # wres may expose pvalue or pvalue attribute in different statmodels versions
                joint_pval = getattr(wres, 'pvalue', None) or getattr(wres, 'pvals', None) or getattr(wres, 'pvalues', None)
                # If pvalue is an array (one per df), take the last element or scalar
                if isinstance(joint_pval, (list, tuple, np.ndarray)):
                    joint_pval = float(np.array(joint_pval).ravel()[-1])
                else:
                    joint_pval = float(joint_pval) if joint_pval is not None else None
            except Exception:
                joint_pval = None
        summary['joint_interaction_pvalue'] = joint_pval

        # Predicted probability change across observed age range for each culture:
        # We will construct new data at observed min and max Age_c in the data used to fit the model,
        # set Age2 = Age_c**2, and set other covariates to their sample means.
        try:
            df_used = res.model.data.frame.copy()
            # Ensure Age_c and Age2 present
            if 'Age_c' in df_used.columns and 'Age2' in df_used.columns:
                age_min = float(df_used['Age_c'].min())
                age_max = float(df_used['Age_c'].max())
                # covariates means/modes
                covars = {}
                for col in df_used.columns:
                    if col in ('Age_c', 'Age2', res.model.endog_names):
                        continue
                    # For categorical culture, we'll vary it below
                    if col == 'culture':
                        continue
                    # use mean for numeric/binary covariates
                    if pd.api.types.is_numeric_dtype(df_used[col]):
                        covars[col] = float(df_used[col].mean())
                    else:
                        # If non-numeric, take the first observed value
                        covars[col] = df_used[col].iloc[0]
                cultures = sorted(pd.unique(df_used['culture']).tolist())
                prob_changes = {}
                for cult in cultures:
                    row_min = {'Age_c': age_min, 'Age2': age_min ** 2, 'culture': cult}
                    row_max = {'Age_c': age_max, 'Age2': age_max ** 2, 'culture': cult}
                    # add covariates
                    for k, v in covars.items():
                        row_min[k] = v
                        row_max[k] = v
                    # Build dataframes; keep column order consistent with model
                    newdf = pd.DataFrame([row_min, row_max])
                    # Ensure all columns expected by the model exist in newdf; if not, attempt to add NaNs
                    for col in res.model.exog_names:
                        # exog_names include the intercept 'Intercept' or 'const' sometimes; statsmodels will handle that
                        if col not in newdf.columns:
                            # If exog is interaction-coded (e.g., dummy columns) we rely on Patsy to build these from formula.
                            # However res.predict accepts a dataframe with original variable names and Patsy will rebuild design matrix.
                            # So do nothing here; missing columns may be handled by Patsy.
                            pass
                    # Predict using model's predict (it will go through Patsy to construct design matrix)
                    try:
                        preds = res.predict(newdf)
                        prob_min = float(preds.iloc[0])
                        prob_max = float(preds.iloc[1])
                        prob_changes[str(cult)] = {
                            'age_c_min': age_min,
                            'age_c_max': age_max,
                            'prob_min': prob_min,
                            'prob_max': prob_max,
                            'delta_prob': prob_max - prob_min
                        }
                    except Exception:
                        # If prediction fails (e.g., mismatched columns), record None
                        prob_changes[str(cult)] = {
                            'age_c_min': age_min,
                            'age_c_max': age_max,
                            'prob_min': None,
                            'prob_max': None,
                            'delta_prob': None
                        }
                summary['predicted_probability_change_across_age_per_culture'] = prob_changes
            else:
                summary['predicted_probability_change_across_age_per_culture'] = None
        except Exception:
            summary['predicted_probability_change_across_age_per_culture'] = None

        return summary

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be the dictionary returned by the modeling function.")

    # Extract and summarize socialcopy_model
    if 'socialcopy_model' in model_output and model_output['socialcopy_model'] is not None:
        try:
            res_sc = model_output['socialcopy_model']
            out['object']['socialcopy_model'] = summarize_result(res_sc)
        except Exception as e:
            out['object']['socialcopy_model'] = {'error': f'Failed to summarize socialcopy_model: {str(e)}'}
    else:
        out['object']['socialcopy_model'] = None

    # Extract and summarize majority_model (might be None if too few copiers)
    if 'majority_model' in model_output and model_output['majority_model'] is not None:
        try:
            res_mc = model_output['majority_model']
            out['object']['majority_model'] = summarize_result(res_mc)
        except Exception as e:
            out['object']['majority_model'] = {'error': f'Failed to summarize majority_model: {str(e)}'}
    else:
        out['object']['majority_model'] = None

    # Short interpretive description to help the user read the numeric output
    out['description'] = (
        "Returned objects contain: (1) a parameter table for each fitted model "
        "(coefficients, standard errors, p-values, and 95% CIs), (2) a focused "
        "listing of Age-related terms (Age_c, Age2) and any Age_c x Culture interaction terms, "
        "(3) whether any interaction terms are individually significant (p < .05), "
        "and (4) a joint Wald-test p-value (if computable) testing whether all Age_c:C(culture) "
        "coefficients are zero. Also included are predicted probability changes across the observed "
        "Age_c range for each culture (holding other covariates at their sample means) so you can "
        "see how the model implies developmental change differs by culture.\n\n"
        "How to read the numbers in relation to the research question:\n"
        "- If the Age_c coefficient (or a combination of Age_c and Age2) in the socialcopy_model "
        "is statistically significant, this indicates an overall developmental change in the tendency "
        "to copy demonstrators vs. choose an undemonstrated option. A positive coefficient for Age_c "
        "means copying increases with age (over the centered age range); a negative one means it decreases.\n"
        "- If any Age_c:C(culture) terms are significant or the joint interaction p-value is < 0.05, "
        "this indicates the developmental trajectory of copying differs across cultural sites.\n"
        "- The predicted_probability_change_across_age_per_culture entries show the model-predicted "
        "probability of the outcome at the observed minimum and maximum Age_c for each culture and the "
        "difference between them (delta_prob). Positive delta_prob means an increase in probability with age "
        "for that culture; negative means a decrease.\n\n"
        "Use the numeric outputs in out['object'] to make formal statements (e.g., 'Age effect is significant, "
        "p = X; Age x Culture joint test p = Y; in culture A predicted probability increases by Z between min and max ages')."
    )

    return out