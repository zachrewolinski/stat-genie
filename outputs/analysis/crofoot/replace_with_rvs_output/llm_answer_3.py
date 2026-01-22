def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, 95% CIs, odds ratios (OR) and
    interpretable marginal effects for the primary predictors:
      - size_ratio
      - FocalHome
      - size_ratio:FocalHome (interaction)

    Returns:
      {
        "object": {
          "terms": {
            "<term>": {
              "coef": float,
              "se": float,
              "pval": float,
              "ci_lower": float,
              "ci_upper": float,
              "odds_ratio": float,
              "or_ci_lower": float,
              "or_ci_upper": float
            }, ...
          },
          "marginal_effects": {
            "size_effect_when_FocalHome_0": { "logit_coef":..., "OR":..., "OR_CI":(...) },
            "size_effect_when_FocalHome_1": { "logit_coef":..., "OR":..., "OR_CI":(...) }
          }
        },
        "description": "text explanation"
      }
    """
    import numpy as np

    res = model_output

    # Helper to safely get series/dict-like attributes
    try:
        params = res.params
    except Exception:
        raise ValueError("Model output has no .params attribute. Provide a statsmodels results object.")
    try:
        bse = res.bse
    except Exception:
        # fallback: try to compute from cov_params if available
        try:
            cov = res.cov_params()
            bse = np.sqrt(np.diag(cov))
            # align index if params is a Series
            if hasattr(params, "index"):
                bse = type(params)(bse, index=params.index)
        except Exception:
            raise ValueError("Cannot obtain standard errors from model output.")
    # p-values
    pvalues = None
    try:
        pvalues = res.pvalues
    except Exception:
        # if pvalues not present, set to NaN
        pvalues = params * 0 + np.nan
        if hasattr(params, "index"):
            pvalues = type(params)(pvalues, index=params.index)

    # conf_int: try provided, otherwise compute normal-approx
    try:
        ci_df = res.conf_int()
        # conf_int returns DataFrame-like with 0,1 columns or similar
        if hasattr(ci_df, "iloc"):
            ci_lower = ci_df.iloc[:, 0]
            ci_upper = ci_df.iloc[:, 1]
        else:
            # fallback if it's array
            ci_lower = ci_df[:, 0]
            ci_upper = ci_df[:, 1]
        if hasattr(params, "index"):
            # try to align index
            try:
                ci_lower = type(params)(ci_lower, index=params.index)
                ci_upper = type(params)(ci_upper, index=params.index)
            except Exception:
                pass
    except Exception:
        # normal approximation
        z = 1.96
        ci_lower = params - z * bse
        ci_upper = params + z * bse

    # function to find exact or partial match for term name in index
    def find_term_index(term_name):
        if hasattr(params, "index"):
            names = list(params.index)
            # exact match
            if term_name in names:
                return term_name
            # try common alternative notations
            alternatives = [
                term_name,
                term_name.replace(":", "*"),  # unlikely but safe
                term_name.replace(":", ":"),
                term_name.replace("FocalHome", "FocalHome[T.1]"),
                term_name.replace("FocalHome", "FocalHome[T.1]"),
            ]
            for alt in alternatives:
                if alt in names:
                    return alt
            # try contains
            for n in names:
                if term_name in str(n):
                    return n
        # if params not indexed or not found
        return None

    # target terms
    main_terms = ["size_ratio", "FocalHome", "size_ratio:FocalHome"]
    extracted = {"terms": {}}

    for t in main_terms:
        idx = find_term_index(t)
        if idx is None:
            extracted["terms"][t] = None
            continue
        coef = float(params[idx])
        se = float(bse[idx])
        pval = float(pvalues[idx]) if idx in pvalues.index else float(pvalues[idx]) if hasattr(pvalues, "__getitem__") else float("nan")
        ci_l = float(ci_lower[idx])
        ci_u = float(ci_upper[idx])
        or_ = float(np.exp(coef))
        or_ci_l = float(np.exp(ci_l))
        or_ci_u = float(np.exp(ci_u))
        extracted["terms"][t] = {
            "coef": coef,
            "se": se,
            "pval": pval,
            "ci_lower": ci_l,
            "ci_upper": ci_u,
            "odds_ratio": or_,
            "or_ci_lower": or_ci_l,
            "or_ci_upper": or_ci_u
        }

    # compute marginal effect of size_ratio when FocalHome = 0 and 1
    # effect_when_FH0 = beta_size
    # effect_when_FH1 = beta_size + beta_interaction
    size_idx = find_term_index("size_ratio")
    int_idx = find_term_index("size_ratio:FocalHome")
    marg = {}
    if size_idx is None:
        marg["size_effect_when_FocalHome_0"] = None
        marg["size_effect_when_FocalHome_1"] = None
    else:
        beta_size = float(params[size_idx])
        se_size = float(bse[size_idx])
        # when FocalHome = 0
        or0 = float(np.exp(beta_size))
        # CI for beta_size
        ci0_l = float(ci_lower[size_idx])
        ci0_u = float(ci_upper[size_idx])
        or0_ci = (float(np.exp(ci0_l)), float(np.exp(ci0_u)))
        marg["size_effect_when_FocalHome_0"] = {
            "logit_coef": beta_size,
            "se": se_size,
            "OR": or0,
            "OR_CI": or0_ci,
            "pval": float(pvalues[size_idx]) if size_idx in pvalues.index else float("nan")
        }

        # when FocalHome = 1
        if int_idx is not None:
            beta_int = float(params[int_idx])
            # variance of sum = var(beta_size) + var(beta_int) + 2*cov(beta_size,beta_int)
            try:
                cov = res.cov_params()
                # cov might be DataFrame; access elements
                var_size = float(cov.loc[size_idx, size_idx])
                var_int = float(cov.loc[int_idx, int_idx])
                cov_si = float(cov.loc[size_idx, int_idx])
                se_sum = float(np.sqrt(var_size + var_int + 2 * cov_si))
                beta_sum = beta_size + beta_int
                or1 = float(np.exp(beta_sum))
                z = 1.96
                ci1_l = beta_sum - z * se_sum
                ci1_u = beta_sum + z * se_sum
                or1_ci = (float(np.exp(ci1_l)), float(np.exp(ci1_u)))
                # compute p-value for sum is not directly available; we keep pval for interaction separately
                marg["size_effect_when_FocalHome_1"] = {
                    "logit_coef": beta_sum,
                    "se": se_sum,
                    "OR": or1,
                    "OR_CI": or1_ci,
                    "pval_interaction": float(pvalues[int_idx]) if int_idx in pvalues.index else float("nan")
                }
            except Exception:
                # fallback without covariance
                beta_sum = beta_size + beta_int
                # approximate se by sqrt(se_size^2 + se_int^2)
                se_int = float(bse[int_idx])
                se_sum = float(np.sqrt(se_size ** 2 + se_int ** 2))
                or1 = float(np.exp(beta_sum))
                z = 1.96
                ci1_l = beta_sum - z * se_sum
                ci1_u = beta_sum + z * se_sum
                or1_ci = (float(np.exp(ci1_l)), float(np.exp(ci1_u)))
                marg["size_effect_when_FocalHome_1"] = {
                    "logit_coef": beta_sum,
                    "se": se_sum,
                    "OR": or1,
                    "OR_CI": or1_ci,
                    "pval_interaction": float(pvalues[int_idx]) if int_idx in pvalues.index else float("nan")
                }
        else:
            marg["size_effect_when_FocalHome_1"] = None

    extracted["marginal_effects"] = marg

    # Build a short description interpreting the key results
    desc_lines = []
    # check existence and significance
    def sig_label(p):
        try:
            if p < 0.001:
                return "*** (p<0.001)"
            elif p < 0.01:
                return "** (p<0.01)"
            elif p < 0.05:
                return "* (p<0.05)"
            else:
                return f"(p={p:.3f})"
        except Exception:
            return "(p=NA)"
    t_size = extracted["terms"].get("size_ratio")
    t_home = extracted["terms"].get("FocalHome")
    t_int = extracted["terms"].get("size_ratio:FocalHome")

    if t_size is not None:
        desc_lines.append(
            f"Relative group size (size_ratio): coef={t_size['coef']:.3f}, OR={t_size['odds_ratio']:.3f}, "
            f"95%CI_OR=({t_size['or_ci_lower']:.3f}, {t_size['or_ci_upper']:.3f}) {sig_label(t_size['pval'])}."
            " Positive coef => larger focal group increases odds of winning."
        )
    else:
        desc_lines.append("Relative group size (size_ratio) term not found in model output.")

    if t_home is not None:
        desc_lines.append(
            f"Focal home advantage (FocalHome): coef={t_home['coef']:.3f}, OR={t_home['odds_ratio']:.3f}, "
            f"95%CI_OR=({t_home['or_ci_lower']:.3f}, {t_home['or_ci_upper']:.3f}) {sig_label(t_home['pval'])}."
            " Positive coef => contests nearer the focal group's home increase odds of focal win."
        )
    else:
        desc_lines.append("FocalHome term not found in model output.")

    if t_int is not None:
        desc_lines.append(
            f"Interaction (size_ratio:FocalHome): coef={t_int['coef']:.3f}, OR={t_int['odds_ratio']:.3f}, "
            f"95%CI_OR=({t_int['or_ci_lower']:.3f}, {t_int['or_ci_upper']:.3f}) {sig_label(t_int['pval'])}."
            " A significant interaction indicates the effect of relative size differs when focal group has home advantage."
        )
    else:
        desc_lines.append("Interaction term size_ratio:FocalHome not found in model output.")

    # Add marginal interpretation if available
    if marg.get("size_effect_when_FocalHome_0") is not None:
        m0 = marg["size_effect_when_FocalHome_0"]
        desc_lines.append(
            f"Marginal effect of size when FocalHome=0: logit_coef={m0['logit_coef']:.3f}, OR={m0['OR']:.3f}."
        )
    if marg.get("size_effect_when_FocalHome_1") is not None:
        m1 = marg["size_effect_when_FocalHome_1"]
        desc_lines.append(
            f"Marginal effect of size when FocalHome=1: logit_coef={m1['logit_coef']:.3f}, OR={m1['OR']:.3f}."
        )

    description = " ".join(desc_lines)

    return {"object": extracted, "description": description}