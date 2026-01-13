def extract_final_answer(model_output):
    """
    Extract statistics relevant to the effect of relative group size (RelSize_Ratio)
    and its interaction with contest location (ContestLoc) from a fitted statsmodels
    GLMResultsWrapper object that was returned by the provided model function.
    
    Returns a dictionary with keys:
      - "object": dict with extracted numeric results (coefficients, SEs, p-values,
                  95% CIs, odds ratios) for RelSize_Ratio main effect and any
                  interaction terms that include RelSize_Ratio; plus a joint
                  Wald test for the interaction terms when present.
      - "description": human-readable explanation of the extracted numbers and how
                       to interpret them for the research question.
    """
    import numpy as np
    import pandas as pd

    out = {"object": None, "description": None}
    try:
        # Prefer cluster-robust results if attached by the model function
        if hasattr(model_output, "cluster_robust") and model_output.cluster_robust is not None:
            res = model_output.cluster_robust
            results_used = "cluster_robust"
        else:
            res = model_output
            results_used = "original"

        # Pull parameter table items
        params = res.params
        pvalues = res.pvalues
        bse = res.bse
        try:
            conf = res.conf_int()
            # conf_int returns DataFrame or ndarray; normalize to DataFrame with matching index
            if not isinstance(conf, pd.DataFrame):
                conf = pd.DataFrame(conf, index=params.index, columns=["2.5%", "97.5%"])
            conf.columns = ["ci_lower", "ci_upper"]
        except Exception:
            # If conf_int not available, compute approx CI using coef +/- 1.96*se
            ci_lower = params - 1.96 * bse
            ci_upper = params + 1.96 * bse
            conf = pd.DataFrame({"ci_lower": ci_lower, "ci_upper": ci_upper})

        # Identify parameters related to RelSize_Ratio (main effect and interactions)
        param_names = list(params.index.astype(str))
        rel_params = [n for n in param_names if "RelSize_Ratio" in n]

        if len(rel_params) == 0:
            out["object"] = {
                "results_used": results_used,
                "message": "No model parameters containing 'RelSize_Ratio' were found in the fitted object.",
            }
            out["description"] = (
                "The fitted model does not expose any parameters with name containing "
                "'RelSize_Ratio'. Ensure the model was fitted with the variable named exactly 'RelSize_Ratio'."
            )
            return out

        # Build detailed table for RelSize_Ratio-related parameters
        rel_table = {}
        significant = []
        for name in rel_params:
            coef = float(params[name])
            se = float(bse[name]) if name in bse.index else None
            p = float(pvalues[name]) if name in pvalues.index else None
            ci_low = float(conf.loc[name, "ci_lower"]) if name in conf.index else None
            ci_up = float(conf.loc[name, "ci_upper"]) if name in conf.index else None
            odds = float(np.exp(coef)) if coef is not None else None
            or_low = float(np.exp(ci_low)) if ci_low is not None else None
            or_up = float(np.exp(ci_up)) if ci_up is not None else None

            rel_table[name] = {
                "coef_log_odds": coef,
                "std_error": se,
                "p_value": p,
                "ci_95_log_odds": (ci_low, ci_up),
                "odds_ratio": odds,
                "ci_95_odds_ratio": (or_low, or_up),
            }
            if p is not None and p < 0.05:
                significant.append(name)

        # Identify strictly interaction parameters (those other than the pure main effect)
        interaction_params = [n for n in rel_params if n != "RelSize_Ratio"]

        # Joint Wald test for the interaction terms (if any)
        interaction_test = None
        if len(interaction_params) > 0:
            # Build restriction R to test that all interaction coefficients = 0
            try:
                indices = [param_names.index(n) for n in interaction_params]
                R = np.zeros((len(indices), len(param_names)))
                for i, j in enumerate(indices):
                    R[i, j] = 1.0
                # wald_test returns a ContrastResults-like object
                wt = res.wald_test(R)
                # Extract statistic and p-value if available
                try:
                    wald_stat = float(wt.statistic)
                except Exception:
                    # sometimes statistic is array-like
                    wald_stat = float(np.array(wt.statistic).squeeze())
                # pvalue attribute may be named pvalue or p_val/p_value depending on version
                wald_p = None
                if hasattr(wt, "pvalue"):
                    wald_p = float(wt.pvalue)
                elif hasattr(wt, "pval"):
                    wald_p = float(wt.pval)
                elif hasattr(wt, "p_value"):
                    wald_p = float(wt.p_value)
                else:
                    wald_p = None

                interaction_test = {
                    "interaction_param_names": interaction_params,
                    "wald_statistic": wald_stat,
                    "wald_p_value": wald_p,
                    "interpretation": (
                        "Small p-value (e.g. < 0.05) indicates that the interaction terms "
                        "are jointly significantly different from zero (i.e. the effect "
                        "of RelSize_Ratio differs by ContestLoc)."
                    ),
                }
            except Exception as e:
                interaction_test = {
                    "error": "Failed to compute joint Wald test for interactions.",
                    "exception": repr(e),
                }

        # Prepare final object
        out_obj = {
            "results_used": results_used,
            "relsize_parameters": rel_table,
            "significant_relsize_parameters": significant,
            "interaction_joint_test": interaction_test,
        }

        # Build a readable description of what these numbers mean for the research question
        desc_lines = []
        desc_lines.append(
            "This output reports the log-odds coefficient, standard error, p-value, "
            "95% confidence interval, and odds-ratio (exp(coef)) for the main effect "
            "of RelSize_Ratio and any interaction terms involving RelSize_Ratio."
        )
        desc_lines.append(
            "Interpretation guidance: a positive log-odds coefficient (or OR>1) for "
            "RelSize_Ratio means that as the focal group's size increases relative to "
            "the other group, the odds that the focal group wins increase. "
            "An interaction term between RelSize_Ratio and ContestLoc that is significant "
            "means that this effect of relative size differs depending on contest location."
        )
        if len(significant) > 0:
            desc_lines.append(
                f"The following RelSize_Ratio-related parameters are statistically significant (p < 0.05): {significant}."
            )
        else:
            desc_lines.append(
                "No RelSize_Ratio-related parameters were statistically significant at p < 0.05."
            )
        if interaction_test is not None:
            if "wald_p_value" in interaction_test and interaction_test["wald_p_value"] is not None:
                if interaction_test["wald_p_value"] < 0.05:
                    desc_lines.append(
                        "The joint Wald test for interaction terms is significant "
                        f"(W={interaction_test['wald_statistic']:.3f}, p={interaction_test['wald_p_value']:.3f}), "
                        "indicating that the effect of relative group size differs by contest location."
                    )
                else:
                    desc_lines.append(
                        "The joint Wald test for interaction terms is not significant "
                        f"(W={interaction_test['wald_statistic']:.3f}, p={interaction_test['wald_p_value']:.3f}), "
                        "indicating no evidence that the effect of relative size differs by location."
                    )
            else:
                desc_lines.append(
                    "A joint test for interactions was attempted but the p-value could not be extracted."
                )

        out["object"] = out_obj
        out["description"] = " ".join(desc_lines)
        return out

    except Exception as exc:
        # In case anything goes wrong, return the exception text to help debugging
        out["object"] = {"error": "exception while extracting results", "exception": repr(exc)}
        out["description"] = (
            "An error occurred while extracting statistics from the provided model output. "
            "See the 'object' element for the exception details."
        )
        return out