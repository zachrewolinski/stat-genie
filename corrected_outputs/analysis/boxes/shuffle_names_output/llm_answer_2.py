def extract_final_answer(model_output):
    """
    Extracts key statistics from a fitted statsmodels GLMResultsWrapper (logistic regression)
    to answer how reliance on majority preference develops with age across cultural contexts.

    Returns a dict with:
      - "object": dict of extracted numerical results (coefficients, SE, z, p, CI, and joint test)
      - "description": brief interpretation of each extracted result in context
    """
    import numpy as np
    import pandas as pd

    res = model_output  # GLMResultsWrapper

    # Basic parameter table
    params = res.params
    bse = res.bse

    # Safely get t- or z-values without triggering ambiguous truth-value checks
    if hasattr(res, "tvalues"):
        tvalues = getattr(res, "tvalues")
    elif hasattr(res, "zvalues"):
        tvalues = getattr(res, "zvalues")
    else:
        tvalues = None

    pvalues = res.pvalues
    conf = res.conf_int()  # DataFrame-like: (lower, upper)
    # Ensure conf has two columns; rename them if possible
    try:
        conf = conf.copy()
        conf.columns = ["CI_lower", "CI_upper"]
    except Exception:
        # If conf_int returned an array-like, convert to DataFrame
        conf = pd.DataFrame(conf, index=params.index, columns=["CI_lower", "CI_upper"])

    # Helper to pull stats for a parameter name (if present)
    def get_param_stats(name):
        if name not in params.index:
            return None
        # Some results objects store statistics as numpy arrays/Series; use .loc safely
        coef = params.loc[name]
        se = bse.loc[name]
        p = pvalues.loc[name]
        ci_lower = conf.loc[name, "CI_lower"]
        ci_upper = conf.loc[name, "CI_upper"]
        z_or_t = None
        if tvalues is not None and name in getattr(tvalues, "index", []):
            z_or_t = tvalues.loc[name]
        # Cast to native Python types for JSON-serializable output
        return {
            "name": name,
            "coef": float(coef) if np.ndim(coef) == 0 else float(np.asarray(coef).item()),
            "se": float(se) if np.ndim(se) == 0 else float(np.asarray(se).item()),
            "z_or_t": float(z_or_t) if (z_or_t is not None and np.ndim(z_or_t) == 0) else (float(np.asarray(z_or_t).item()) if z_or_t is not None else None),
            "p": float(p) if np.ndim(p) == 0 else float(np.asarray(p).item()),
            "ci_lower": float(ci_lower) if np.ndim(ci_lower) == 0 else float(np.asarray(ci_lower).item()),
            "ci_upper": float(ci_upper) if np.ndim(ci_upper) == 0 else float(np.asarray(ci_upper).item()),
        }

    # Extract main developmental predictors
    age_main = get_param_stats("Age_c")
    age_quad = get_param_stats("Age_sq")

    # Extract interaction terms (Age_c x Site).
    # Look for any parameter name that contains 'Age_c' but is not exactly 'Age_c'.
    interaction_names = [n for n in params.index if ("Age_c" in str(n) and str(n) != "Age_c")]
    # Sort for stable output
    interaction_names = sorted(interaction_names)

    interactions = []
    for n in interaction_names:
        stats = get_param_stats(n)
        if stats is not None:
            interactions.append(stats)

    # Joint Wald test for all interaction coefficients = 0 (tests whether age effect differs across sites)
    joint_test = None
    if len(interaction_names) > 0:
        # Build R matrix selecting the interaction parameters
        idx_map = {name: i for i, name in enumerate(params.index)}
        R = np.zeros((len(interaction_names), len(params)))
        for i, name in enumerate(interaction_names):
            R[i, idx_map[name]] = 1.0
        # Perform Wald test
        try:
            wt = res.wald_test(R)
            # Depending on statsmodels version, wt may expose statistic as array-like or a scalar
            stat_arr = getattr(wt, "statistic", None)
            if stat_arr is None:
                # some versions return a tuple-like; try calling as array
                stat_arr = np.atleast_1d(wt)
            stat = float(np.atleast_1d(stat_arr)[0])
            # pvalue access
            pval = None
            if hasattr(wt, "pvalue"):
                pval = float(np.atleast_1d(getattr(wt, "pvalue"))[0])
            elif hasattr(wt, "pvalues"):
                pval = float(np.atleast_1d(getattr(wt, "pvalues"))[0])
            else:
                # try attribute-like access
                try:
                    pval = float(wt.p)  # unlikely, but attempt
                except Exception:
                    pval = None
            # degrees of freedom: try common attributes, fallback to number of tests
            df = None
            if hasattr(wt, "df_denom") and wt.df_denom is not None:
                try:
                    df = int(getattr(wt, "df_denom"))
                except Exception:
                    df = None
            if df is None and hasattr(wt, "df"):
                try:
                    df = int(getattr(wt, "df"))
                except Exception:
                    df = None
            if df is None:
                df = int(len(interaction_names))
            joint_test = {"chi2": stat, "p": pval, "df": df, "n_terms": len(interaction_names)}
        except Exception as e:
            # Fall back to reporting that the joint test failed
            joint_test = {"error": f"wald_test failed: {e}", "n_terms": len(interaction_names)}

    # Compose the return object
    object_out = {
        "age_main": age_main,
        "age_quadratic": age_quad,
        "age_by_site_interactions": interactions,
        "interactions_joint_test": joint_test,
        "note": "Model is a logistic GLM predicting probability of choosing majority. Coefficients are on the log-odds scale."
    }

    # Short interpretation in context
    # We avoid asserting significance automatically; user can inspect p-values returned above.
    description_lines = []
    if age_main is not None:
        description_lines.append(
            "Age main effect (Age_c): positive coef -> higher log-odds of choosing majority with increasing age; "
            "negative coef -> lower log-odds. See object['age_main']."
        )
    else:
        description_lines.append("Age main effect (Age_c) not found in model output.")

    if age_quad is not None:
        description_lines.append(
            "Age quadratic (Age_sq) captures curvature in the age trajectory (quadratic effect). "
            "A significant coef indicates nonlinear change with age. See object['age_quadratic']."
        )
    else:
        description_lines.append("Age quadratic (Age_sq) not found in model output.")

    if interactions:
        description_lines.append(
            "Age x Site interaction terms are listed in object['age_by_site_interactions']. "
            "Each shows how the slope of Age_c differs for that site versus the reference site. "
            "Check their p-values to identify which sites differ from the reference."
        )
        if joint_test is not None and ("p" in joint_test) and (joint_test.get("p") is not None):
            try:
                if joint_test["p"] < 0.05:
                    description_lines.append(
                        f"A joint Wald test (chi2={joint_test['chi2']:.3f}, df={joint_test['df']}, p={joint_test['p']:.3g}) "
                        "indicates that age-related change differs across sites (significant moderation)."
                    )
                else:
                    description_lines.append(
                        f"A joint Wald test (chi2={joint_test['chi2']:.3f}, df={joint_test['df']}, p={joint_test['p']:.3g}) "
                        "does NOT provide evidence that age-related change differs across sites."
                    )
            except Exception:
                description_lines.append("Joint test result could not be formatted for description.")
        else:
            description_lines.append("Joint test for interactions not available.")
    else:
        description_lines.append("No Age x Site interaction terms found in the model output; age effect assumed constant across sites in this fitted model.")

    description = " ".join(description_lines)

    return {"object": object_out, "description": description}