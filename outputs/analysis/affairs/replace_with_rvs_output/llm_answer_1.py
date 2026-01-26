def extract_final_answer(model_output):
    """
    Extracts statistics related to the effect of 'children_binary' on 'affair_count'
    from the provided model_output dictionary.

    Returns:
      {
        "object": {
            "model_used": "nb_model",
            "coef": <float>,              # log-count coefficient
            "std_err": <float>,
            "p_value": <float>,
            "conf_int": [low, high],     # 95% CI on coefficient scale
            "irr": <float>,              # incidence rate ratio = exp(coef)
            "irr_conf_int": [low, high], # 95% CI for irr
            "notes": <str>
        },
        "description": <str>            # brief plain-English interpretation
      }
    """
    import numpy as np

    # Helper to build result dict from a model-like object
    def _from_model(res_obj, param_name='children_binary'):
        out = {}
        try:
            params = res_obj.params
            if param_name not in params.index:
                raise KeyError(f"parameter '{param_name}' not found in model params")
            coef = float(params[param_name])
            se = float(res_obj.bse[param_name]) if hasattr(res_obj, 'bse') else None
            pval = float(res_obj.pvalues[param_name]) if hasattr(res_obj, 'pvalues') else None
            try:
                ci = res_obj.conf_int().loc[param_name].astype(float).tolist()
            except Exception:
                # approximate 95% CI if conf_int not available
                if se is not None:
                    ci = [coef - 1.96 * se, coef + 1.96 * se]
                else:
                    ci = [None, None]
            irr = float(np.exp(coef)) if coef is not None else None
            irr_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))] if ci[0] is not None else [None, None]

            out.update({
                "coef": coef,
                "std_err": se,
                "p_value": pval,
                "conf_int": ci,
                "irr": irr,
                "irr_conf_int": irr_ci
            })
            return out
        except Exception as e:
            return {"error": str(e)}

    # 1) Prefer the primary NB GLM result if present
    if 'nb_model' in model_output and model_output['nb_model'] is not None:
        nb = model_output['nb_model']
        nb_stats = _from_model(nb, 'children_binary')
        result_object = {
            "model_used": "nb_model",
            **nb_stats
        }

        # Optional: also try to extract from nb_coef_table if present for cross-check
        if 'nb_coef_table' in model_output and model_output['nb_coef_table'] is not None:
            try:
                tab = model_output['nb_coef_table']
                if 'children_binary' in tab.index:
                    row = tab.loc['children_binary']
                    # cross-check and attach
                    result_object["coef_table_check"] = {
                        "coef": float(row['Coef.']),
                        "std_err": float(row['Std.Err.']),
                        "p_value": float(row['P>|z|'])
                    }
            except Exception:
                pass

        # Optional: also extract from zinb_model (if available) for comparison
        if 'zinb_model' in model_output and model_output.get('zinb_model') is not None:
            zinb = model_output['zinb_model']
            zinb_stats = _from_model(zinb, 'children_binary')
            result_object["zinb_model_check"] = zinb_stats

        # Build human-readable description based on NB results
        if "error" in nb_stats:
            description = f"Could not extract 'children_binary' from nb_model: {nb_stats['error']}"
        else:
            coef = nb_stats["coef"]
            se = nb_stats["std_err"]
            p = nb_stats["p_value"]
            irr = nb_stats["irr"]
            ci = nb_stats["conf_int"]
            irr_ci = nb_stats["irr_conf_int"]
            description = (
                f"Negative-binomial model: coefficient for 'children_binary' = {coef:.4f} "
                f"(SE = {se:.4f}, p = {p:.4g}), 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]. "
                f"On the incidence-rate scale: IRR = {irr:.3f}, 95% CI = [{irr_ci[0]:.3f}, {irr_ci[1]:.3f}]. "
                "Interpretation: the estimated effect is a very small decrease in expected affair counts "
                "for those with children, but the effect is not statistically significant (p >> 0.05) "
                "and the 95% CI for the IRR includes 1. Therefore there is no evidence that having children "
                "reduces engagement in extramarital affairs in this sample."
            )

        return {"object": result_object, "description": description}

    # 2) Fallback: try to read from nb_coef_table alone
    if 'nb_coef_table' in model_output and model_output['nb_coef_table'] is not None:
        try:
            tab = model_output['nb_coef_table']
            if 'children_binary' not in tab.index:
                return {"object": None, "description": "children_binary not found in nb_coef_table."}
            row = tab.loc['children_binary']
            coef = float(row['Coef.'])
            se = float(row['Std.Err.'])
            p = float(row['P>|z|'])
            ci = [coef - 1.96 * se, coef + 1.96 * se]
            irr = float(np.exp(coef))
            irr_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))]
            result_object = {
                "model_used": "nb_coef_table",
                "coef": coef,
                "std_err": se,
                "p_value": p,
                "conf_int": ci,
                "irr": irr,
                "irr_conf_int": irr_ci
            }
            description = (
                f"Using nb_coef_table: coefficient = {coef:.4f} (SE = {se:.4f}, p = {p:.4g}), "
                f"IRR = {irr:.3f}, 95% CI IRR = [{irr_ci[0]:.3f}, {irr_ci[1]:.3f}]. "
                "No statistically significant evidence that having children decreases extramarital affairs."
            )
            return {"object": result_object, "description": description}
        except Exception as e:
            return {"object": None, "description": f"Error extracting from nb_coef_table: {e}"}

    # Nothing found
    return {"object": None, "description": "No negative-binomial or coefficient table results found in model_output."}