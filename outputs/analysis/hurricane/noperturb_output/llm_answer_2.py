def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of the independent variable 'masfem_z'
    from the model_output returned by the modeling function.

    Returns a dictionary with keys:
      - "object": dict with extracted stats for each fitted model (or None)
      - "description": human-readable explanation of the extracted stats (or errors and remediation)

    The function handles:
      - statsmodels result objects under keys 'nb_model' and 'ols_damage'
      - error messages under 'nb_model_error' and 'ols_damage_error'
      - absence of models
    """
    import math
    out = {"object": None, "description": ""}

    extracted = {}
    errors = {}

    if not isinstance(model_output, dict):
        out["description"] = "model_output is not a dict. Expected the dict returned by the modeling function."
        return out

    # Helper to extract from a statsmodels result object
    def _extract_from_result(res, varname="masfem_z", is_count_model=False):
        info = {}
        try:
            params = getattr(res, "params", None)
            pvalues = getattr(res, "pvalues", None)
            conf = None
            try:
                conf = res.conf_int()
            except Exception:
                conf = None

            # locate the variable name in case it was transformed/named differently
            if params is None or varname not in params.index:
                # try to find a parameter name containing varname
                candidates = [n for n in params.index] if params is not None else []
                matches = [n for n in candidates if varname in n]
                if len(matches) == 1:
                    var = matches[0]
                elif len(matches) > 1:
                    var = matches[0]
                else:
                    raise KeyError(f"Variable '{varname}' not found in model parameters: {list(params.index) if params is not None else 'no params'}")
            else:
                var = varname

            coef = float(params[var])
            pval = float(pvalues[var]) if (pvalues is not None and var in pvalues.index) else None
            ci_lower, ci_upper = (None, None)
            if conf is not None and var in conf.index:
                ci_lower, ci_upper = float(conf.loc[var, 0]), float(conf.loc[var, 1])

            info["var"] = var
            info["coef"] = coef
            info["pvalue"] = pval
            info["ci_lower"] = ci_lower
            info["ci_upper"] = ci_upper

            if is_count_model:
                # For count model (negative binomial GLM with log link), exponentiate coef -> incidence rate ratio (IRR)
                try:
                    irr = math.exp(coef)
                    info["irr"] = irr
                    if ci_lower is not None and ci_upper is not None:
                        info["irr_ci_lower"] = math.exp(ci_lower)
                        info["irr_ci_upper"] = math.exp(ci_upper)
                except Exception:
                    pass
            else:
                # For OLS on logged damage: approximate percent change = 100*(exp(coef)-1)
                try:
                    pct_change = (math.exp(coef) - 1) * 100.0
                    info["pct_change_approx"] = pct_change
                    if ci_lower is not None and ci_upper is not None:
                        info["pct_change_ci_lower"] = (math.exp(ci_lower) - 1) * 100.0
                        info["pct_change_ci_upper"] = (math.exp(ci_upper) - 1) * 100.0
                except Exception:
                    pass

            # significance flag
            info["significant_p_lt_0.05"] = (pval is not None and pval < 0.05)
            return info
        except Exception as e:
            raise

    # Negative binomial model
    if "nb_model" in model_output and model_output["nb_model"] is not None:
        nb = model_output["nb_model"]
        try:
            nb_info = _extract_from_result(nb, varname="masfem_z", is_count_model=True)
            extracted["nb_model"] = nb_info
        except Exception as e:
            errors["nb_model_extract_error"] = str(e)
    elif "nb_model_error" in model_output and model_output["nb_model_error"]:
        errors["nb_model_error"] = str(model_output["nb_model_error"])

    # OLS damage model
    if "ols_damage" in model_output and model_output["ols_damage"] is not None:
        ols = model_output["ols_damage"]
        try:
            ols_info = _extract_from_result(ols, varname="masfem_z", is_count_model=False)
            extracted["ols_damage"] = ols_info
        except Exception as e:
            errors["ols_damage_extract_error"] = str(e)
    elif "ols_damage_error" in model_output and model_output["ols_damage_error"]:
        errors["ols_damage_error"] = str(model_output["ols_damage_error"])
    elif "ols_damage" in model_output and model_output["ols_damage"] is None:
        # explicitly not fitted due to insufficient data; treat as no model
        errors["ols_damage"] = "OLS damage model was not fit (None returned)."

    # Build description
    desc_lines = []
    if extracted:
        desc_lines.append("Extracted statistics for 'masfem_z':")
        for key, info in extracted.items():
            if key == "nb_model":
                line = (
                    f"- Negative Binomial (deaths): parameter '{info['var']}' coef = {info['coef']:.4f}, "
                    f"p = {info['pvalue']:.4g}" + (
                        f", 95% CI = [{info['ci_lower']:.4f}, {info['ci_upper']:.4f}]" if info.get("ci_lower") is not None else ""
                    ) + (
                        f". IRR = {info['irr']:.4f}" +
                        (f" (95% CI [{info.get('irr_ci_lower', float('nan')):.4f}, {info.get('irr_ci_upper', float('nan')):.4f}])" if info.get('irr_ci_lower') is not None else "")
                        if info.get("irr") is not None else ""
                    ) + (
                        f". Significant at p<0.05: {info['significant_p_lt_0.05']}"
                    )
                )
                desc_lines.append(line)
            elif key == "ols_damage":
                line = (
                    f"- OLS on log(damage): parameter '{info['var']}' coef = {info['coef']:.4f}, p = {info['pvalue']:.4g}" + (
                        f", 95% CI = [{info['ci_lower']:.4f}, {info['ci_upper']:.4f}]" if info.get("ci_lower") is not None else ""
                    ) + (
                        f". Approx. percent change in damage per unit masfem_z = {info.get('pct_change_approx', float('nan')):.2f}%"
                        + (f" (95% CI [{info.get('pct_change_ci_lower', float('nan')):.2f}%, {info.get('pct_change_ci_upper', float('nan')):.2f}%])" if info.get('pct_change_ci_lower') is not None else "")
                    ) + (
                        f". Significant at p<0.05: {info['significant_p_lt_0.05']}"
                    )
                )
                desc_lines.append(line)
        out["object"] = extracted
    else:
        out["object"] = None
        desc_lines.append("No fitted models with extractable statistics were found.")

    if errors:
        desc_lines.append("Errors or issues encountered:")
        for k, msg in errors.items():
            desc_lines.append(f"- {k}: {msg}")

        # Provide a likely cause and a suggested fix for the specific error seen in the original run:
        # Many recent runs fail with "Cannot interpret 'Int64Dtype()' as a data type" from statsmodels when pandas uses nullable integer dtype.
        desc_lines.append(
            "Likely cause: statsmodels does not accept pandas' nullable integer dtype ('Int64') for model matrices. "
            "If you see messages like \"Cannot interpret 'Int64Dtype()' as a data type\", convert integer columns to standard numpy dtypes "
            "(int64 or float64) and categorical columns to strings or pandas category dtype before fitting.\n"
            "Suggested quick fixes (run before modeling):\n"
            "  df['deaths'] = df['deaths'].astype(float)\n"
            "  df['masfem_z'] = df['masfem_z'].astype(float)\n"
            "  df['wind_z'] = df['wind_z'].astype(float)\n"
            "  df['category_z'] = df['category_z'].astype(float)\n"
            "  df['min_pressure_z'] = df['min_pressure_z'].astype(float)\n"
            "  df['year_z'] = df['year_z'].astype(float)\n"
            "  df['elapsedyrs_z'] = df['elapsedyrs_z'].astype(float)\n"
            "  df['IsFemaleName'] = df['IsFemaleName'].astype(int)\n"
            "  df['source'] = df['source'].astype(str)\n"
            "Then re-run the modeling function. After models fit, re-call this extractor on the returned output."
        )

    out["description"] = "\n".join(desc_lines)
    return out