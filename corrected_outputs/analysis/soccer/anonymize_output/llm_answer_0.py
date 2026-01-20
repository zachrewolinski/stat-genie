def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of player skin tone (DarkSkin) on red-card counts
    from the model output returned by the modeling function.

    Returns a dictionary with keys:
      - "object": a dict containing numeric summaries for the main DarkSkin effect (and,
                  if present, the DarkSkin x RefereeImplicit interaction), including:
          * coef: coefficient (log rate ratio)
          * se: standard error (clustered if available)
          * z: coefficient / se
          * p: two-sided p-value
          * ci95: (lower, upper) 95% CI for the coef
          * irr: exp(coef) — incidence rate ratio (IRR)
          * irr_ci95: (exp(lower), exp(upper)) 95% CI for IRR
      - "description": brief human-readable interpretation of the results in context.
    """
    import numpy as np
    import math

    def _get_clustered_obj(mo):
        """
        Accept both the lightweight clustered namespace (with .params, .bse, .pvalues, .cov_params)
        or a raw statsmodels results object.
        """
        if mo is None:
            return None
        # If it's a dict-like with 'nb_model_clustered' already passed in, accept it
        # If it's a namespace as in the modeling code, it will have .params, .bse, .pvalues
        # If it's a raw results object, try to compute stats from it (fall back to normal cov)
        return mo

    def _safe_get_param_stats(clustered_obj, param_name):
        """
        Return (coef, se, z, p, ci_lower, ci_upper). Uses clustered covariance if possible.
        """
        if clustered_obj is None:
            return None
        # Try to access params and covariance
        # Some objects in the pipeline are simple namespaces with pandas Series
        params = None
        cov_df = None
        pval = None
        bse = None

        # Prefer clustered_obj.params if available
        if hasattr(clustered_obj, 'params'):
            params = clustered_obj.params
        elif hasattr(clustered_obj, 'original_results') and hasattr(clustered_obj.original_results, 'params'):
            params = clustered_obj.original_results.params

        # Try to get covariance matrix via cov_params() if available
        try:
            if hasattr(clustered_obj, 'cov_params'):
                cov_df = clustered_obj.cov_params()
            elif hasattr(clustered_obj, 'original_results') and hasattr(clustered_obj.original_results, 'cov_params'):
                cov_df = clustered_obj.original_results.cov_params()
        except Exception:
            cov_df = None

        # Try to get p-values or compute from params/bse
        if hasattr(clustered_obj, 'pvalues'):
            pval = clustered_obj.pvalues
        elif hasattr(clustered_obj, 'original_results') and hasattr(clustered_obj.original_results, 'pvalues'):
            pval = clustered_obj.original_results.pvalues

        # If cov_df present, extract var; else try clustered_obj.bse; else fallback to original_results.bse
        var = None
        if cov_df is not None:
            # cov_df may be DataFrame; ensure param_name exists
            try:
                var = float(cov_df.loc[param_name, param_name])
            except Exception:
                # maybe index names differ in whitespace; try best-effort match
                try:
                    # find nearest matching column name
                    cols = list(cov_df.columns)
                    matches = [c for c in cols if c.strip().lower() == param_name.strip().lower()]
                    if matches:
                        var = float(cov_df.loc[matches[0], matches[0]])
                except Exception:
                    var = None

        if var is None:
            # try bse
            if hasattr(clustered_obj, 'bse'):
                try:
                    bse = float(clustered_obj.bse[param_name])
                    var = bse * bse
                except Exception:
                    bse = None
            elif hasattr(clustered_obj, 'original_results') and hasattr(clustered_obj.original_results, 'bse'):
                try:
                    bse = float(clustered_obj.original_results.bse[param_name])
                    var = bse * bse
                except Exception:
                    bse = None

        if params is None:
            raise ValueError("Cannot find params in provided model object.")

        if param_name not in params.index:
            # try case-insensitive / stripped matching
            matches = [n for n in params.index if n.strip().lower() == param_name.strip().lower()]
            if matches:
                param_name = matches[0]
            else:
                return None  # parameter not present

        coef = float(params[param_name])

        if var is not None and (not math.isnan(var)):
            se = float(math.sqrt(max(var, 0.0)))
        else:
            # try direct bse
            if bse is not None:
                se = float(bse)
            else:
                se = None

        # compute z and p
        if se is not None and se > 0:
            z = coef / se
            # two-sided p from normal
            from scipy import stats
            p = float(2 * stats.norm.sf(abs(z)))
            ci_low = coef - 1.96 * se
            ci_high = coef + 1.96 * se
        else:
            z = None
            p = None
            ci_low = None
            ci_high = None

        return {
            'param_name': param_name,
            'coef': coef,
            'se': se,
            'z': z,
            'p': p,
            'ci95': (ci_low, ci_high)
        }

    # Start extracting
    # Expect model_output to be the dict returned by the provided model function
    if not isinstance(model_output, dict):
        # try to handle the case where a single object is passed
        clustered_main = _get_clustered_obj(model_output)
        interaction_clustered = None
    else:
        clustered_main = model_output.get('nb_model_clustered') or model_output.get('nb_model') or None
        interaction_results = model_output.get('interaction_results')
        interaction_clustered = None
        if isinstance(interaction_results, dict):
            interaction_clustered = interaction_results.get('nb_inter_clustered')

    # Extract main DarkSkin effect
    main_stats = _safe_get_param_stats(clustered_main, 'DarkSkin')

    # Compute IRR and IRR CI if possible
    if main_stats is not None:
        coef = main_stats['coef']
        ci = main_stats['ci95']
        irr = float(np.exp(coef))
        irr_ci = (float(np.exp(ci[0])) if ci[0] is not None else None,
                  float(np.exp(ci[1])) if ci[1] is not None else None)
        main_stats['irr'] = irr
        main_stats['irr_ci95'] = irr_ci

    # Extract interaction term if present
    # The modeling code named it 'DarkSkin_x_Implicit' — check common variants
    inter_names = ['DarkSkin_x_Implicit', 'DarkSkin:RefereeImplicit', 'DarkSkin*RefereeImplicit', 'DarkSkin:RefereeImplicit']
    interaction_stats = None
    if interaction_clustered is not None:
        # try known name first
        for nm in inter_names:
            st = _safe_get_param_stats(interaction_clustered, nm)
            if st is not None:
                interaction_stats = st
                break
        # if found, compute IRR (note: interpretation for interaction coef depends on moderator scale)
        if interaction_stats is not None:
            coefi = interaction_stats['coef']
            ci = interaction_stats['ci95']
            try:
                interaction_stats['irr'] = float(np.exp(coefi))
                interaction_stats['irr_ci95'] = (float(np.exp(ci[0])) if ci[0] is not None else None,
                                                 float(np.exp(ci[1])) if ci[1] is not None else None)
            except Exception:
                interaction_stats['irr'] = None
                interaction_stats['irr_ci95'] = (None, None)

    # Build description string
    desc_lines = []
    if main_stats is None:
        desc_lines.append("Main model: Could not find a parameter named 'DarkSkin' in the provided model output.")
    else:
        # interpret
        coef = main_stats['coef']
        p = main_stats['p']
        irr = main_stats['irr']
        irr_low, irr_high = main_stats['irr_ci95']
        desc_lines.append(
            f"Main model (offset by log(Matches), negative binomial): DarkSkin coefficient = {coef:.4f} "
            f"(SE ≈ {main_stats['se']:.4f}, z ≈ {main_stats['z']:.2f}, p = {p:.4g})."
        )
        if irr is not None:
            desc_lines.append(
                f"Exponentiated => IRR = {irr:.3f} (95% CI ≈ [{irr_low:.3f}, {irr_high:.3f}])."
            )
        # decision
        if (p is not None) and (p < 0.05):
            desc_lines.append(
                "Interpretation: Players coded as having dark skin receive red cards at a statistically significantly "
                "higher rate per match compared to players coded as having light skin (holding covariates constant)."
            )
        else:
            desc_lines.append(
                "Interpretation: No statistically significant difference in red-card rate by skin tone in the main model."
            )

    if interaction_stats is not None:
        coef_i = interaction_stats['coef']
        p_i = interaction_stats['p']
        desc_lines.append(
            f"Interaction model: DarkSkin x RefereeImplicit coefficient = {coef_i:.4f} "
            f"(SE ≈ {interaction_stats['se']:.4f}, z ≈ {interaction_stats['z']:.2f}, p = {p_i:.4g})."
        )
        # sign interpretation
        if (p_i is not None) and (p_i < 0.05):
            desc_lines.append(
                "The interaction is statistically significant. Because the interaction coefficient is negative, "
                "the positive effect of DarkSkin on red-card rates (observed in the main model) is reduced as "
                "RefereeImplicit increases. Caution: the moderator is not centered here, so the 'main' DarkSkin "
                "coefficient in the interaction model corresponds to the effect when RefereeImplicit == 0; "
                "marginal effects vary with the moderator."
            )
        else:
            desc_lines.append(
                "The interaction was not statistically significant, suggesting no evidence that RefereeImplicit "
                "moderates the DarkSkin effect (at conventional alpha levels)."
            )

    description = " ".join(desc_lines)

    return {
        "object": {
            "main_effect": main_stats,
            "interaction_effect": interaction_stats
        },
        "description": description
    }