def extract_final_answer(model_output):
    """
    Extracts coefficients, SEs, p-values, confidence intervals, and simple interpretations
    for the 'beauty_z' (linear) and 'beauty_z_sq' (quadratic) terms from a fitted
    statsmodels RegressionResultsWrapper.

    Returns:
      {
        "object": {
            "coef_beauty": float,
            "se_beauty": float,
            "t_beauty": float,
            "p_beauty": float,
            "ci_beauty": (float_lower, float_upper),
            "coef_beauty_sq": float,
            "se_beauty_sq": float,
            "t_beauty_sq": float,
            "p_beauty_sq": float,
            "ci_beauty_sq": (float_lower, float_upper),
            "marginal_at_0": float,         # derivative at beauty_z = 0 (== coef_beauty)
            "marginal_at_+1": float,        # derivative at beauty_z = +1 SD
            "marginal_at_-1": float,        # derivative at beauty_z = -1 SD
            "vertex_x": float or None,      # location of extremum in SD units, if defined
            "vertex_change": float or None  # predicted change in eval relative to 0 at vertex: b1*x + b2*x^2
        },
        "description": "<concise interpretation string>"
      }
    """
    import numpy as np

    # Basic checks
    try:
        params = model_output.params
        bse = model_output.bse
        tvalues = model_output.tvalues
        pvalues = model_output.pvalues
        ci = model_output.conf_int()
    except Exception as e:
        raise ValueError("model_output does not look like a statsmodels results object or is missing attributes") from e

    # Ensure the expected parameter names exist
    for name in ('beauty_z', 'beauty_z_sq'):
        if name not in params.index:
            raise ValueError(f"Expected parameter '{name}' not found in model_output.params")

    # Extract statistics for linear term
    b1 = float(params['beauty_z'])
    se1 = float(bse['beauty_z'])
    t1 = float(tvalues['beauty_z'])
    p1 = float(pvalues['beauty_z'])
    try:
        ci1 = tuple(ci.loc['beauty_z'].astype(float))
    except Exception:
        # fall back if ci is an ndarray
        ci = np.asarray(ci)
        idx = list(params.index).index('beauty_z')
        ci1 = (float(ci[idx, 0]), float(ci[idx, 1]))

    # Extract statistics for quadratic term
    b2 = float(params['beauty_z_sq'])
    se2 = float(bse['beauty_z_sq'])
    t2 = float(tvalues['beauty_z_sq'])
    p2 = float(pvalues['beauty_z_sq'])
    try:
        ci2 = tuple(ci.loc['beauty_z_sq'].astype(float))
    except Exception:
        ci = np.asarray(ci)
        idx = list(params.index).index('beauty_z_sq')
        ci2 = (float(ci[idx, 0]), float(ci[idx, 1]))

    # Marginal effects: derivative dy/dx = b1 + 2*b2*x
    marginal_at_0 = b1
    marginal_at_plus1 = b1 + 2 * b2 * 1.0
    marginal_at_minus1 = b1 + 2 * b2 * (-1.0)

    # Vertex (extremum) location x* = -b1 / (2*b2) if b2 != 0
    if abs(b2) > 1e-12:
        vertex_x = -b1 / (2 * b2)
        # predicted change in eval (relative to beauty_z=0) at vertex: b1*x + b2*x^2
        vertex_change = b1 * vertex_x + b2 * (vertex_x ** 2)
        vertex_x = float(vertex_x)
        vertex_change = float(vertex_change)
    else:
        vertex_x = None
        vertex_change = None

    # Simple significance summary
    sig_thresh = 0.05
    linear_sig = p1 < sig_thresh
    quad_sig = p2 < sig_thresh

    # Direction and shape interpretation
    if quad_sig:
        shape = "concave (inverted-U)" if b2 < 0 else "convex (U-shaped)"
    else:
        shape = "no strong evidence of curvature" if not quad_sig else ""

    # Compose a concise description
    desc_lines = []
    desc_lines.append(
        f"Linear term (beauty_z): coef={b1:.4f}, SE={se1:.4f}, t={t1:.3f}, p={p1:.3g}, 95% CI=({ci1[0]:.4f}, {ci1[1]:.4f})."
    )
    desc_lines.append(
        f"Quadratic term (beauty_z_sq): coef={b2:.4f}, SE={se2:.4f}, t={t2:.3f}, p={p2:.3g}, 95% CI=({ci2[0]:.4f}, {ci2[1]:.4f})."
    )
    desc_lines.append(
        f"Marginal effect at mean beauty (z=0): {marginal_at_0:.4f} change in evaluation points per +1 SD in attractiveness."
    )
    desc_lines.append(
        f"Marginal effects at z=+1: {marginal_at_plus1:.4f}; at z=-1: {marginal_at_minus1:.4f}."
    )
    if vertex_x is not None:
        desc_lines.append(
            f"Estimated extremum at beauty_z = {vertex_x:.3f} SDs (relative change in eval at that point = {vertex_change:.4f})."
        )
    desc_lines.append(
        f"Interpretation: {'Evidence' if (linear_sig or quad_sig) else 'No strong evidence'} that beauty affects course evaluations."
    )
    if quad_sig:
        desc_lines.append(f"The quadratic term is significant, suggesting a {shape}.")
    else:
        if linear_sig:
            desc_lines.append("The effect appears linear (significant linear term, nonsignificant quadratic).")
        else:
            desc_lines.append("Neither linear nor quadratic terms are statistically significant at p<0.05.")

    description = " ".join(desc_lines)

    output_object = {
        "coef_beauty": b1,
        "se_beauty": se1,
        "t_beauty": t1,
        "p_beauty": p1,
        "ci_beauty": ci1,
        "coef_beauty_sq": b2,
        "se_beauty_sq": se2,
        "t_beauty_sq": t2,
        "p_beauty_sq": p2,
        "ci_beauty_sq": ci2,
        "marginal_at_0": marginal_at_0,
        "marginal_at_+1": marginal_at_plus1,
        "marginal_at_-1": marginal_at_minus1,
        "vertex_x": vertex_x,
        "vertex_change": vertex_change,
        "linear_significant": bool(linear_sig),
        "quadratic_significant": bool(quad_sig)
    }

    return {"object": output_object, "description": description}