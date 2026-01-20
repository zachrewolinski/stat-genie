def extract_final_answer(model_output):
    """
    Extract relevant statistics from a fitted statsmodels GLMResults (logit) object
    that relates the probability the focal group wins ('dyad') to:
      - rel_size_z (relative group size, standardized)
      - rel_home_adv_z (home-range advantage, standardized)
      - their interaction
    and controls for n_focal and n_other.

    Returns a dictionary with keys:
      - "object": dict of extracted numeric results (coef table, odds ratios,
                  marginal effects for rel_size at several home-adv values and
                  vice versa, predicted probabilities for a small grid).
      - "description": brief explanation of what the values mean and how to
                       interpret them in the context of the task.
    """
    import numpy as np
    from scipy.stats import norm
    from scipy.special import expit

    res = model_output

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not look like a fitted statsmodels results object.")

    # Names we expect in the model
    expected = ['const', 'rel_size_z', 'rel_home_adv_z', 'interaction', 'n_focal', 'n_other']
    exog_names = list(res.model.exog_names)
    for name in ['rel_size_z', 'rel_home_adv_z', 'interaction', 'n_focal', 'n_other']:
        if name not in exog_names:
            raise KeyError(f"Expected predictor '{name}' not found in model exog names: {exog_names}")

    params = res.params
    bse = res.bse
    pvals = res.pvalues
    conf = res.conf_int()  # 2-column array or DataFrame (idx x [low, high])
    cov = res.cov_params()

    # Build coefficient table
    coef_table = {}
    for name in exog_names:
        coef_table[name] = {
            "coef": float(params[name]),
            "se": float(bse[name]),
            "z": float(params[name] / bse[name]) if bse[name] > 0 else None,
            "pvalue": float(pvals[name]),
            "ci_2.5%": float(conf.loc[name, 0]) if hasattr(conf, "loc") else float(conf[exog_names.index(name), 0]),
            "ci_97.5%": float(conf.loc[name, 1]) if hasattr(conf, "loc") else float(conf[exog_names.index(name), 1]),
            "odds_ratio": float(np.exp(params[name]))
        }

    # Marginal effects for rel_size_z at representative rel_home_adv_z values (-1, 0, 1)
    def marginal_effect_of_rel_size(home_adv_value):
        # effect on log-odds = b_rel_size + b_interaction * home_adv_value
        b1 = params['rel_size_z']
        b3 = params['interaction']
        eff = b1 + b3 * home_adv_value

        # Variance of eff = Var(b1) + c^2 Var(b3) + 2*c Cov(b1,b3)
        var_b1 = cov.loc['rel_size_z', 'rel_size_z'] if hasattr(cov, "loc") else cov[exog_names.index('rel_size_z'), exog_names.index('rel_size_z')]
        var_b3 = cov.loc['interaction', 'interaction'] if hasattr(cov, "loc") else cov[exog_names.index('interaction'), exog_names.index('interaction')]
        cov_b1b3 = cov.loc['rel_size_z', 'interaction'] if hasattr(cov, "loc") else cov[exog_names.index('rel_size_z'), exog_names.index('interaction')]
        var_eff = var_b1 + (home_adv_value ** 2) * var_b3 + 2 * home_adv_value * cov_b1b3
        se_eff = float(np.sqrt(var_eff)) if var_eff >= 0 else None

        z = float(eff / se_eff) if se_eff and se_eff > 0 else None
        p = float(2 * (1 - norm.cdf(abs(z)))) if z is not None else None
        ci_low = eff - 1.96 * se_eff if se_eff is not None else None
        ci_high = eff + 1.96 * se_eff if se_eff is not None else None

        return {
            "home_adv_value": home_adv_value,
            "log_odds_effect_per_sd_rel_size": float(eff),
            "se": se_eff,
            "z": z,
            "pvalue": p,
            "ci_logodds": [ci_low, ci_high] if ci_low is not None else None,
            "odds_ratio_per_sd_rel_size": float(np.exp(eff)),
            "odds_ratio_ci": [float(np.exp(ci_low)), float(np.exp(ci_high))] if ci_low is not None else None
        }

    marginal_rel_size_at_home = [marginal_effect_of_rel_size(c) for c in [-1.0, 0.0, 1.0]]

    # Marginal effects for rel_home_adv_z at representative rel_size_z values (-1, 0, 1)
    def marginal_effect_of_home(rel_size_value):
        # effect on log-odds = b_rel_home + b_interaction * rel_size_value
        b2 = params['rel_home_adv_z']
        b3 = params['interaction']
        eff = b2 + b3 * rel_size_value

        var_b2 = cov.loc['rel_home_adv_z', 'rel_home_adv_z'] if hasattr(cov, "loc") else cov[exog_names.index('rel_home_adv_z'), exog_names.index('rel_home_adv_z')]
        var_b3 = cov.loc['interaction', 'interaction'] if hasattr(cov, "loc") else cov[exog_names.index('interaction'), exog_names.index('interaction')]
        cov_b2b3 = cov.loc['rel_home_adv_z', 'interaction'] if hasattr(cov, "loc") else cov[exog_names.index('rel_home_adv_z'), exog_names.index('interaction')]
        var_eff = var_b2 + (rel_size_value ** 2) * var_b3 + 2 * rel_size_value * cov_b2b3
        se_eff = float(np.sqrt(var_eff)) if var_eff >= 0 else None

        z = float(eff / se_eff) if se_eff and se_eff > 0 else None
        p = float(2 * (1 - norm.cdf(abs(z)))) if z is not None else None
        ci_low = eff - 1.96 * se_eff if se_eff is not None else None
        ci_high = eff + 1.96 * se_eff if se_eff is not None else None

        return {
            "rel_size_value": rel_size_value,
            "log_odds_effect_per_sd_home_adv": float(eff),
            "se": se_eff,
            "z": z,
            "pvalue": p,
            "ci_logodds": [ci_low, ci_high] if ci_low is not None else None,
            "odds_ratio_per_sd_home_adv": float(np.exp(eff)),
            "odds_ratio_ci": [float(np.exp(ci_low)), float(np.exp(ci_high))] if ci_low is not None else None
        }

    marginal_home_at_rel_size = [marginal_effect_of_home(c) for c in [-1.0, 0.0, 1.0]]

    # Predicted probabilities for a small grid of rel_size_z x rel_home_adv_z values,
    # holding n_focal and n_other at their sample means (from the model's exog)
    exog = np.asarray(res.model.exog)
    # Find column indices for n_focal and n_other in exog_names
    idx_n_focal = exog_names.index('n_focal')
    idx_n_other = exog_names.index('n_other')
    mean_n_focal = float(np.mean(exog[:, idx_n_focal]))
    mean_n_other = float(np.mean(exog[:, idx_n_other]))

    grid = []
    for rs in [-1.0, 0.0, 1.0]:
        for rh in [-1.0, 0.0, 1.0]:
            interaction = rs * rh
            # construct design vector in the same order as exog_names
            xvec = np.zeros(len(exog_names))
            for i, nm in enumerate(exog_names):
                if nm == 'const':
                    xvec[i] = 1.0
                elif nm == 'rel_size_z':
                    xvec[i] = rs
                elif nm == 'rel_home_adv_z':
                    xvec[i] = rh
                elif nm == 'interaction':
                    xvec[i] = interaction
                elif nm == 'n_focal':
                    xvec[i] = mean_n_focal
                elif nm == 'n_other':
                    xvec[i] = mean_n_other
                else:
                    xvec[i] = 0.0  # fallback

            # predicted logit and probability
            logit = float(np.dot(params.values if hasattr(params, "values") else params, xvec))
            prob = float(expit(logit))

            # compute se of linear predictor: Var(x'B) = x' Cov(B) x
            cov_mat = np.asarray(cov)  # works whether DataFrame or ndarray
            var_logit = float(np.dot(xvec, np.dot(cov_mat, xvec)))
            se_logit = np.sqrt(var_logit) if var_logit >= 0 else None
            ci_logit = (logit - 1.96 * se_logit, logit + 1.96 * se_logit) if se_logit is not None else (None, None)
            ci_prob = (float(expit(ci_logit[0])), float(expit(ci_logit[1]))) if se_logit is not None else (None, None)

            grid.append({
                "rel_size_z": rs,
                "rel_home_adv_z": rh,
                "mean_n_focal": mean_n_focal,
                "mean_n_other": mean_n_other,
                "logit": logit,
                "probability_focal_wins": prob,
                "logit_se": se_logit,
                "prob_ci": ci_prob
            })

    # Summarize significance of the primary terms for a quick-answer style
    def significance_label(p):
        if p < 0.001:
            return "p < 0.001"
        elif p < 0.01:
            return "p < 0.01"
        elif p < 0.05:
            return "p < 0.05"
        else:
            return f"p = {p:.3f}"

    primary_summary = {
        "rel_size_z": {
            "coef": float(params['rel_size_z']),
            "odds_ratio": float(np.exp(params['rel_size_z'])),
            "pvalue": float(pvals['rel_size_z']),
            "sig": significance_label(float(pvals['rel_size_z']))
        },
        "rel_home_adv_z": {
            "coef": float(params['rel_home_adv_z']),
            "odds_ratio": float(np.exp(params['rel_home_adv_z'])),
            "pvalue": float(pvals['rel_home_adv_z']),
            "sig": significance_label(float(pvals['rel_home_adv_z']))
        },
        "interaction": {
            "coef": float(params['interaction']),
            "odds_ratio": float(np.exp(params['interaction'])),
            "pvalue": float(pvals['interaction']),
            "sig": significance_label(float(pvals['interaction']))
        }
    }

    result_object = {
        "coef_table": coef_table,
        "primary_summary": primary_summary,
        "marginal_rel_size_at_home": marginal_rel_size_at_home,
        "marginal_home_at_rel_size": marginal_home_at_rel_size,
        "predicted_probability_grid": grid
    }

    description = (
        "This output contains coefficient estimates, standard errors, z-statistics, p-values, 95% CIs, "
        "and odds ratios for all model terms. The 'primary_summary' gives a concise view of the three "
        "terms of interest (rel_size_z, rel_home_adv_z, and their interaction) and their significance. "
        "'marginal_rel_size_at_home' shows how a one-standard-deviation change in relative group size "
        "affects the log-odds and odds of the focal group winning when the contest location advantage is "
        "at -1, 0, and +1 SDs (i.e., away from focal, neutral, and close to focal). Similarly, "
        "'marginal_home_at_rel_size' shows how a one-SD change in home-range advantage affects outcomes at "
        "rel_size = -1, 0, +1 SDs. 'predicted_probability_grid' gives predicted probabilities (with CIs) for combinations "
        "of rel_size_z and rel_home_adv_z (each at -1, 0, +1 SD), holding n_focal and n_other at their sample means. "
        "Interpretation guidance: a positive coefficient increases the focal group's log-odds of winning; an "
        "odds ratio > 1 indicates higher odds of focal winning per 1-SD increase in that predictor. Significant "
        "p-values (e.g., p < 0.05) indicate evidence that the predictor is associated with win probability. "
        "The interaction term (if significant) means the effect of relative group size depends on contest location (and vice versa)."
    )

    return {"object": result_object, "description": description}