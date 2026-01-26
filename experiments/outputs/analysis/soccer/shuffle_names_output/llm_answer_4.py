def extract_final_answer(model_output):
    """
    Extracts the key statistics for the SkinToneDark coefficient from the model output.

    Returns a dict with:
      - "object": dict with numeric results (coef, pvalue, IRR, IRR_CI_lower, IRR_CI_upper, significant)
      - "description": short plain-language interpretation in context
    """
    import numpy as np

    # Prepare default empty result in case extraction fails
    result_obj = {
        "coef": None,
        "pvalue": None,
        "IRR": None,
        "IRR_CI_lower": None,
        "IRR_CI_upper": None,
        "significant": None
    }

    try:
        # Prefer using the pre-computed irr_table if present
        irr_table = model_output.get('irr_table', None)

        if irr_table is not None:
            # irr_table might be a pandas DataFrame or a dict-like structure
            try:
                # If it's a DataFrame-like object
                coef = float(irr_table.loc['SkinToneDark', 'coef'])
                irr = float(irr_table.loc['SkinToneDark', 'IRR'])
                irr_lo = float(irr_table.loc['SkinToneDark', 'IRR_2.5%'])
                irr_hi = float(irr_table.loc['SkinToneDark', 'IRR_97.5%'])
                # p-value may not be in irr_table; retrieve from model_result if available
                model_res = model_output.get('model_result', None)
                if model_res is not None and hasattr(model_res, 'pvalues'):
                    pval = float(model_res.pvalues.get('SkinToneDark', np.nan))
                else:
                    pval = float(np.nan)
            except Exception:
                # fallback: maybe irr_table is a dict or different layout
                if isinstance(irr_table, dict):
                    # try to locate index of SkinToneDark
                    keys = list(irr_table.get('coef', {}).keys())
                    if 'SkinToneDark' in keys:
                        coef = float(irr_table['coef']['SkinToneDark'])
                        irr = float(irr_table['IRR'][keys.index('SkinToneDark')])
                        irr_lo = float(irr_table['IRR_2.5%'][keys.index('SkinToneDark')])
                        irr_hi = float(irr_table['IRR_97.5%'][keys.index('SkinToneDark')])
                    else:
                        raise
                    model_res = model_output.get('model_result', None)
                    pval = float(model_res.pvalues.get('SkinToneDark', np.nan)) if model_res is not None else float(np.nan)
                else:
                    raise

        else:
            # If irr_table not provided, extract from model_result directly
            model_res = model_output.get('model_result', None)
            if model_res is None:
                raise ValueError("No model_result or irr_table found in model_output.")

            params = model_res.params
            coef = float(params['SkinToneDark'])
            # p-value (may reflect clustered SEs if model_res was clustered)
            pval = float(model_res.pvalues['SkinToneDark']) if hasattr(model_res, 'pvalues') else float(np.nan)
            conf = model_res.conf_int().loc['SkinToneDark']
            irr = float(np.exp(coef))
            irr_lo = float(np.exp(conf[0]))
            irr_hi = float(np.exp(conf[1]))

        # Fill result object
        result_obj["coef"] = coef
        result_obj["pvalue"] = pval
        result_obj["IRR"] = irr
        result_obj["IRR_CI_lower"] = irr_lo
        result_obj["IRR_CI_upper"] = irr_hi
        # Declare significance at conventional alpha=0.05 if p-value available
        result_obj["significant"] = (pval < 0.05) if (pval is not None and not np.isnan(pval)) else None

    except Exception as e:
        # If extraction fails, include the exception message in description below
        desc = f"Failed to extract SkinToneDark results: {e}"
        return {"object": result_obj, "description": desc}

    # Build a concise interpretation
    if result_obj["IRR"] is not None:
        ir_percent = (result_obj["IRR"] - 1.0) * 100.0
        sig_text = "statistically significant" if result_obj["significant"] else "not statistically significant"
        description = (
            f"Negative binomial model (offset = log(Matches), controls included) estimate for SkinToneDark: "
            f"coef = {result_obj['coef']:.4f}, p = {result_obj['pvalue']:.3f}. "
            f"Incidence rate ratio (IRR) = {result_obj['IRR']:.3f} "
            f"(95% CI: {result_obj['IRR_CI_lower']:.3f} – {result_obj['IRR_CI_upper']:.3f}). "
            f"Interpretation: players rated as having a dark skin tone receive approximately {ir_percent:.1f}% "
            f"more red cards than lighter-skinned players, {sig_text} at alpha=0.05."
        )
    else:
        description = "Could not compute IRR for SkinToneDark from the provided model_output."

    return {"object": result_obj, "description": description}