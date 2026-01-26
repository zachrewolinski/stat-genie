def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of name femininity on hurricane fatalities
    from a dictionary of fitted statsmodels result objects.

    Input:
      model_output: dict where values are statsmodels fitted result wrappers
                    (or strings/errors). Expected keys include:
                    'neg_binomial', 'poisson', 'ols_log_deaths',
                    'neg_binomial_mturk', 'ols_log_deaths_mturk',
                    'neg_binomial_interaction', etc.

    Output (dict):
      {
        "object": {
          "models": {
            "<model_key>": {
              "variable": "<var_name>",
              "coef": float,
              "se": float,
              "stat": float,            # z or t
              "pvalue": float,
              "ci_lower": float,
              "ci_upper": float,
              "exp_coef": float/null,   # IRR for count models
              "exp_ci_lower": float/null,
              "exp_ci_upper": float/null,
              "percent_change": float/null # for log-OLS: (exp(beta)-1)*100
            },
            ...
          },
          "interaction_margins": {    # only if interaction model present
            "masfem_z_when_male": { ... },   # effect when gender_mf=0
            "masfem_z_when_female": { ... }  # effect when gender_mf=1
          },
          "support_summary": {
            "n_models_considered": int,
            "n_supporting_hypothesis": int,
            "models_supporting": [list of model keys],
            "conclusion": str
          }
        },
        "description": "Explanation of the above numbers and final interpretation."
      }
    """
    import numpy as np

    # Helper to safely check if an object appears to be a fitted model
    def is_fitted_result(x):
        return hasattr(x, "params") and hasattr(x, "bse") and hasattr(x, "pvalues")

    def get_confint(model):
        try:
            ci = model.conf_int()
            # conf_int can be a DataFrame or ndarray; convert to dict-like lookup
            return ci
        except Exception:
            return None

    def extract_from_model(model, varname):
        """Return dict of stats for varname if present, else None."""
        params = model.params
        if varname not in params.index:
            return None
        coef = float(params.loc[varname])
        # robust se already used in fitting; bse should be aligned
        se = float(model.bse.loc[varname]) if varname in model.bse.index else None
        # compute statistic
        stat = (coef / se) if (se is not None and se != 0) else None
        pval = float(model.pvalues.loc[varname]) if varname in model.pvalues.index else None
        # confidence interval
        ci_raw = get_confint(model)
        if ci_raw is not None:
            try:
                # Try DataFrame-like access
                ci_lower = float(ci_raw.loc[varname][0])
                ci_upper = float(ci_raw.loc[varname][1])
            except Exception:
                try:
                    # ndarray-like: find param position
                    idx = list(params.index).index(varname)
                    ci_lower = float(ci_raw[idx, 0])
                    ci_upper = float(ci_raw[idx, 1])
                except Exception:
                    ci_lower = ci_upper = None
        else:
            ci_lower = ci_upper = None

        # Determine model type: GLM with family (count models) or OLS
        is_glm = hasattr(model, "model") and hasattr(model.model, "family")
        is_ols = hasattr(model, "model") and model.model.__class__.__name__.lower().startswith("ols")

        entry = {
            "variable": varname,
            "coef": coef,
            "se": se,
            "stat": stat,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "exp_coef": None,
            "exp_ci_lower": None,
            "exp_ci_upper": None,
            "percent_change": None
        }

        # For GLM count models with log link (Poisson, NegBin) exponentiate to get IRR
        if is_glm:
            try:
                fam_name = model.model.family.__class__.__name__.lower()
            except Exception:
                fam_name = ""
            # assume log-link count model for Poisson and NegativeBinomial
            if "poisson" in fam_name or "negativebinomial" in fam_name or "negative_binomial" in fam_name:
                try:
                    entry["exp_coef"] = float(np.exp(coef))
                    if (ci_lower is not None) and (ci_upper is not None):
                        entry["exp_ci_lower"] = float(np.exp(ci_lower))
                        entry["exp_ci_upper"] = float(np.exp(ci_upper))
                except Exception:
                    entry["exp_coef"] = entry["exp_ci_lower"] = entry["exp_ci_upper"] = None
        # For OLS on log-deaths: interpret coef as approx log-change; convert to percent change
        if is_ols:
            try:
                pct = (np.exp(coef) - 1.0) * 100.0
                entry["percent_change"] = float(pct)
                if (ci_lower is not None) and (ci_upper is not None):
                    entry["percent_change_ci_lower"] = float((np.exp(ci_lower) - 1.0) * 100.0)
                    entry["percent_change_ci_upper"] = float((np.exp(ci_upper) - 1.0) * 100.0)
            except Exception:
                entry["percent_change"] = None

        return entry

    # Variables to check
    primary_vars = ["masfem_z", "masfem_mturk_z"]
    results = {"models": {}, "interaction_margins": {}, "support_summary": {}}

    # Iterate through provided models
    for key, val in model_output.items():
        if not is_fitted_result(val):
            # skip error strings or messages; could record them
            continue
        model = val
        # Primary var for this model: prefer masfem_z unless the model obviously uses masfem_mturk_z
        # Check both primary vars present in model.params
        found_any = False
        for v in primary_vars:
            if v in model.params.index:
                entry = extract_from_model(model, v)
                if entry is not None:
                    results["models"][key] = entry
                    found_any = True
                    break
        # Special case: interaction model may have 'masfem_z' and 'masfem_z:gender_mf'
        if (not found_any) and ("interaction" in key or ":" in " ".join(model.params.index)):
            # try masfem_z presence
            if "masfem_z" in model.params.index:
                entry = extract_from_model(model, "masfem_z")
                if entry is not None:
                    results["models"][key] = entry
            # also try to extract interaction term if present
            inter_name = "masfem_z:gender_mf"
            if inter_name in model.params.index:
                # store separately under model key + "_interaction"
                inter_entry = extract_from_model(model, inter_name)
                results["models"][key + "_interaction_term"] = inter_entry

                # compute marginal effects for gender_mf = 0 and 1 if masfem_z present
                if "masfem_z" in model.params.index:
                    main = model.params.loc["masfem_z"]
                    inter = model.params.loc[inter_name]
                    # standard errors for margin require delta method; approximate via variance:
                    # var(a+b)=var(a)+var(b)+2cov(a,b). Use model.cov_params() if available.
                    try:
                        cov = model.cov_params()
                        var_main = cov.loc["masfem_z", "masfem_z"]
                        var_inter = cov.loc[inter_name, inter_name]
                        covar = cov.loc["masfem_z", inter_name]
                    except Exception:
                        var_main = var_inter = covar = None

                    # margin when male (gender_mf=0): just main
                    margin_male = {"coef": float(main)}
                    # margin when female (gender_mf=1): main + inter
                    margin_female_coef = float(main + inter)
                    # compute SEs if covariances available
                    if var_main is not None:
                        se_male = float(np.sqrt(var_main))
                        margin_male["se"] = se_male
                    if (var_main is not None) and (var_inter is not None) and (covar is not None):
                        se_female = float(np.sqrt(var_main + var_inter + 2.0 * covar))
                        margin_female = {"coef": margin_female_coef, "se": se_female}
                    else:
                        margin_female = {"coef": margin_female_coef, "se": None}

                    # For GLM count models, provide exp transformation
                    is_glm = hasattr(model, "model") and hasattr(model.model, "family")
                    if is_glm:
                        try:
                            margin_male["exp_coef"] = float(np.exp(margin_male["coef"]))
                            margin_female["exp_coef"] = float(np.exp(margin_female["coef"]))
                        except Exception:
                            pass
                    results["interaction_margins"]["masfem_z_when_male_in_" + key] = margin_male
                    results["interaction_margins"]["masfem_z_when_female_in_" + key] = margin_female

    # Summarize support for hypothesis:
    # Hypothesis: higher femininity -> fewer fatalities.
    # For count models: support if exp_coef < 1 and p < .05
    # For log-OLS: support if percent_change < 0 and p < .05
    considered = 0
    supporting = 0
    supporting_models = []
    for mk, info in results["models"].items():
        # skip interaction_term entries (they will be separate); focus on primary masfem vars
        var = info.get("variable", "")
        if var not in ("masfem_z", "masfem_mturk_z"):
            continue
        considered += 1
        p = info.get("pvalue", None)
        is_signif = (p is not None) and (p < 0.05)
        supported = False
        if info.get("exp_coef", None) is not None:
            # count model
            expc = info["exp_coef"]
            if (expc < 1.0) and is_signif:
                supported = True
        elif info.get("percent_change", None) is not None:
            pct = info["percent_change"]
            if (pct < 0) and is_signif:
                supported = True
        else:
            # fallback: negative coef & significant (conservative)
            coef = info.get("coef", None)
            if (coef is not None) and (coef < 0) and is_signif:
                supported = True

        if supported:
            supporting += 1
            supporting_models.append(mk)

    # Write summary
    if considered == 0:
        conclusion = "No fitted models with masfem variables were found in model_output."
    else:
        results["support_summary"]["n_models_considered"] = considered
        results["support_summary"]["n_supporting_hypothesis"] = supporting
        results["support_summary"]["models_supporting"] = supporting_models
        if supporting / considered >= 0.5:
            conclusion = (
                f"A majority of models ({supporting}/{considered}) show a statistically "
                "significant association in the expected direction (more feminine name -> fewer deaths)."
            )
        elif supporting == 0:
            conclusion = (
                f"None of the {considered} models provide statistically significant evidence that more feminine "
                "names are associated with fewer deaths."
            )
        else:
            conclusion = (
                f"Some models ({supporting}/{considered}) support the hypothesis, but the majority do not; "
                "evidence is mixed."
            )
        results["support_summary"]["conclusion"] = conclusion

    # Final return object and short description
    description_lines = [
        "This output contains extracted coefficients, standard errors, test statistics, p-values, and ",
        "95% confidence intervals for the name-femininity variables (masfem_z and masfem_mturk_z) ",
        "from each fitted model found in model_output. For count models (Negative Binomial, Poisson) ",
        "the exponentiated coefficient (IRR) and its CI are provided; IRR < 1 implies fewer expected deaths ",
        "as name femininity increases. For the OLS on log(deaths) the coefficient is converted to a percent ",
        "change in deaths (exp(beta)-1) * 100. The 'support_summary' reports how many models show a ",
        "statistically significant effect in the hypothesized (negative) direction and a brief conclusion."
    ]
    description = " ".join(description_lines) + " Final conclusion: " + (results["support_summary"].get("conclusion", conclusion))

    return {"object": results, "description": description}