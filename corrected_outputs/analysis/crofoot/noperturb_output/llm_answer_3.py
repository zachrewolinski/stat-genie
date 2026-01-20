def extract_final_answer(model_output):
    """
    Extract key statistics from the RobustResults-like object returned by the model function.
    Returns a dict with:
      - "object": a dict containing coefficients, standard errors, z-values, p-values,
                  odds ratios (with 95% CI) for the main terms and the interaction,
                  and the conditional effect of RelGroupSize_z when ContestLocation==1
                  (i.e., contest closer to the focal group's center).
      - "description": a brief plain-language interpretation of those results.
    The function expects model_output to have attributes:
      - params (pd.Series indexed by parameter names)
      - bse (pd.Series)
      - pvalues (pd.Series)
      - tvalues (pd.Series) or similar (z-values)
      - cov_params() method returning covariance matrix as a DataFrame (or 2D array)
    """
    import numpy as np
    from scipy.stats import norm

    # Names expected in the model
    names = {
        'rel': 'RelGroupSize_z',
        'loc': 'ContestLocation',
        'int': 'RelGroupSize_x_Location'
    }

    # Helper to safely get element from series-like
    def _get(series, name):
        try:
            return series[name]
        except Exception:
            # try positional fallback if name not present
            return series.iloc[list(series.index).index(name)] if name in series.index else None

    # Pull parameters and statistics
    params = getattr(model_output, 'params', None)
    bse = getattr(model_output, 'bse', None)
    pvalues = getattr(model_output, 'pvalues', None)
    tvalues = getattr(model_output, 'tvalues', None)

    if params is None or bse is None or pvalues is None:
        raise ValueError("model_output must provide params, bse, and pvalues attributes.")

    # Extract coefficients and SEs
    beta_rel = _get(params, names['rel'])
    beta_loc = _get(params, names['loc'])
    beta_int = _get(params, names['int'])

    se_rel = _get(bse, names['rel'])
    se_loc = _get(bse, names['loc'])
    se_int = _get(bse, names['int'])

    p_rel = _get(pvalues, names['rel'])
    p_loc = _get(pvalues, names['loc'])
    p_int = _get(pvalues, names['int'])

    z_rel = _get(tvalues, names['rel']) if tvalues is not None else (beta_rel / se_rel if se_rel else None)
    z_loc = _get(tvalues, names['loc']) if tvalues is not None else (beta_loc / se_loc if se_loc else None)
    z_int = _get(tvalues, names['int']) if tvalues is not None else (beta_int / se_int if se_int else None)

    # Obtain covariance matrix for linear-combination inference
    try:
        cov = model_output.cov_params()
        cov_df = cov if hasattr(cov, 'loc') else None
    except Exception:
        cov_df = None

    # Compute conditional effect of RelGroupSize_z when ContestLocation == 1:
    # effect = beta_rel + beta_int
    effect_when_closer = None
    se_when_closer = None
    z_when_closer = None
    p_when_closer = None
    if beta_rel is not None and beta_int is not None:
        effect_when_closer = beta_rel + beta_int
        # compute standard error using covariance if available
        if cov_df is not None:
            try:
                v_rr = cov_df.loc[names['rel'], names['rel']]
                v_ii = cov_df.loc[names['int'], names['int']]
                v_ri = cov_df.loc[names['rel'], names['int']]
                var_sum = v_rr + v_ii + 2.0 * v_ri
                se_when_closer = float(np.sqrt(var_sum))
            except Exception:
                se_when_closer = None
        # fallback to naive combination of SEs if covariance not available (conservative: ignore covariance)
        if se_when_closer is None and se_rel is not None and se_int is not None:
            se_when_closer = float(np.sqrt(se_rel**2 + se_int**2))
        if se_when_closer is not None:
            z_when_closer = effect_when_closer / se_when_closer if se_when_closer != 0 else None
            p_when_closer = float(2 * norm.sf(abs(z_when_closer))) if z_when_closer is not None else None

    # Compute odds ratios and 95% CI for relevant estimates
    def or_and_ci(coef, se):
        if coef is None or se is None:
            return {'OR': None, 'CI95': (None, None)}
        lci = coef - 1.96 * se
        uci = coef + 1.96 * se
        return {'OR': float(np.exp(coef)), 'CI95': (float(np.exp(lci)), float(np.exp(uci)))}

    or_rel = or_and_ci(beta_rel, se_rel)
    or_loc = or_and_ci(beta_loc, se_loc)
    or_int = or_and_ci(beta_int, se_int)
    or_when_closer = or_and_ci(effect_when_closer, se_when_closer) if effect_when_closer is not None else {'OR': None, 'CI95': (None, None)}

    # Build result object to return
    result_object = {
        'RelGroupSize_z': {
            'coef': float(beta_rel) if beta_rel is not None else None,
            'se': float(se_rel) if se_rel is not None else None,
            'z': float(z_rel) if z_rel is not None else None,
            'p': float(p_rel) if p_rel is not None else None,
            'odds_ratio': or_rel['OR'],
            'odds_ratio_95CI': or_rel['CI95']
        },
        'ContestLocation': {
            'coef': float(beta_loc) if beta_loc is not None else None,
            'se': float(se_loc) if se_loc is not None else None,
            'z': float(z_loc) if z_loc is not None else None,
            'p': float(p_loc) if p_loc is not None else None,
            'odds_ratio': or_loc['OR'],
            'odds_ratio_95CI': or_loc['CI95']
        },
        'Interaction_RelGroupSize_x_Location': {
            'coef': float(beta_int) if beta_int is not None else None,
            'se': float(se_int) if se_int is not None else None,
            'z': float(z_int) if z_int is not None else None,
            'p': float(p_int) if p_int is not None else None,
            'odds_ratio': or_int['OR'],
            'odds_ratio_95CI': or_int['CI95']
        },
        'RelGroupSize_effect_when_contest_closer_to_focal': {
            # This is the marginal effect of a one SD increase in RelGroupSize_z on log-odds
            # when ContestLocation == 1 (i.e., contest closer to focal group's center).
            'coef': float(effect_when_closer) if effect_when_closer is not None else None,
            'se': float(se_when_closer) if se_when_closer is not None else None,
            'z': float(z_when_closer) if z_when_closer is not None else None,
            'p': float(p_when_closer) if p_when_closer is not None else None,
            'odds_ratio': or_when_closer['OR'],
            'odds_ratio_95CI': or_when_closer['CI95']
        },
        # include full param vector for reference
        'all_params': {k: float(v) for k, v in dict(params).items()}
    }

    # Short interpretation based on p-values (alpha = 0.05); be cautious if p-values are None
    def sig_label(p):
        if p is None:
            return 'unknown'
        return 'significant' if p < 0.05 else 'not significant'

    rel_sig = sig_label(p_rel)
    loc_sig = sig_label(p_loc)
    int_sig = sig_label(p_int)
    rel_when_closer_sig = sig_label(p_when_closer)

    # Compose human-readable description
    description_lines = []
    description_lines.append(
        f"RelGroupSize_z (relative group size) coefficient = {result_object['RelGroupSize_z']['coef']:.3f}, "
        f"p = {result_object['RelGroupSize_z']['p']:.3g} ({rel_sig}). "
        f"OR = {result_object['RelGroupSize_z']['odds_ratio']:.3f} "
        f"(95% CI {result_object['RelGroupSize_z']['odds_ratio_95CI']})."
    )
    description_lines.append(
        f"ContestLocation (contest closer to focal group's center) coefficient = {result_object['ContestLocation']['coef']:.3f}, "
        f"p = {result_object['ContestLocation']['p']:.3g} ({loc_sig}). "
        f"OR = {result_object['ContestLocation']['odds_ratio']:.3f} "
        f"(95% CI {result_object['ContestLocation']['odds_ratio_95CI']})."
    )
    description_lines.append(
        f"Interaction (RelGroupSize_z x ContestLocation) coefficient = {result_object['Interaction_RelGroupSize_x_Location']['coef']:.3f}, "
        f"p = {result_object['Interaction_RelGroupSize_x_Location']['p']:.3g} ({int_sig}). "
        f"OR = {result_object['Interaction_RelGroupSize_x_Location']['odds_ratio']:.3f} "
        f"(95% CI {result_object['Interaction_RelGroupSize_x_Location']['odds_ratio_95CI']})."
    )
    description_lines.append(
        f"Conditional effect of RelGroupSize_z when contest is closer to focal group's center: "
        f"log-odds = {result_object['RelGroupSize_effect_when_contest_closer_to_focal']['coef']:.3f}, "
        f"p = {result_object['RelGroupSize_effect_when_contest_closer_to_focal']['p']:.3g} ({rel_when_closer_sig}). "
        f"OR = {result_object['RelGroupSize_effect_when_contest_closer_to_focal']['odds_ratio']:.3f} "
        f"(95% CI {result_object['RelGroupSize_effect_when_contest_closer_to_focal']['odds_ratio_95CI']})."
    )

    # Final summary sentence based on significance of main effect and interaction
    if p_rel is not None and p_int is not None:
        if p_int < 0.05:
            summary_sentence = (
                "There is evidence that the effect of relative group size on winning depends on contest location "
                "(significant interaction). Interpret the conditional effects above: the effect when contests are "
                "closer to the focal group's center is given explicitly."
            )
        else:
            if p_rel < 0.05:
                summary_sentence = (
                    "Relative group size has a significant main effect on the probability of the focal group winning, "
                    "and there is no strong evidence that this effect differs by contest location (interaction not significant)."
                )
            else:
                summary_sentence = (
                    "No strong evidence that relative group size affects the probability of winning (main effect not significant), "
                    "and the interaction with contest location is also not significant."
                )
    else:
        summary_sentence = "Unable to determine statistical significance for some terms (missing p-values). See the extracted numbers."

    description_lines.append(summary_sentence)
    description = " ".join(description_lines)

    return {"object": result_object, "description": description}