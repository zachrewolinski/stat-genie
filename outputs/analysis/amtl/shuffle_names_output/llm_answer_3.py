def extract_final_answer(model_output):
    """
    Extract statistics relating to the effect of Genus (with Homo sapiens as reference)
    on AMTL frequency from a statsmodels GLMResultsWrapper.

    Returns a dict with:
      - "object": a dict keyed by each non-reference genus containing:
            - coef: log-odds coefficient (genus vs reference)
            - se: standard error of coef
            - p: p-value for coef (Wald test)
            - ci_lower, ci_upper: 95% confidence interval on the log-odds scale
            - odds_ratio: exp(coef)
            - odds_ratio_ci: [exp(ci_lower), exp(ci_upper)]
        Additionally, if available, an omnibus Wald test for the Genus term is returned under
        the key "_genus_omnibus" with fields 'statistic' and 'p'.
      - "description": short interpretation of the extracted statistics.
    """
    import re
    import numpy as np

    res = model_output

    params = res.params
    pvalues = res.pvalues
    bse = res.bse
    conf = res.conf_int()

    genus_results = {}
    for name in params.index:
        # match parameter names created by statsmodels for treatment-coded categorical:
        # examples:
        #   C(Genus, Treatment(reference="Homo sapiens"))[T.Pan]
        #   C(Genus, Treatment(reference='Homo sapiens'))[T.Pongo]
        m = re.search(r'\[T\.(.+)\]$', name)
        if m:
            level = m.group(1)
            coef = float(params[name])
            se = float(bse[name]) if name in bse.index else None
            p = float(pvalues[name]) if name in pvalues.index else None
            # confidence interval (may raise if missing; guard)
            if name in conf.index:
                ci_lower, ci_upper = float(conf.loc[name, 0]), float(conf.loc[name, 1])
            else:
                ci_lower, ci_upper = None, None

            odds_ratio = float(np.exp(coef))
            or_ci = [float(np.exp(ci_lower)) if ci_lower is not None else None,
                     float(np.exp(ci_upper)) if ci_upper is not None else None]

            genus_results[level] = {
                'coef': coef,
                'se': se,
                'p': p,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'odds_ratio': odds_ratio,
                'odds_ratio_ci': or_ci
            }

    # Try to obtain an omnibus (joint) test for the Genus term (Wald test), if available.
    try:
        wtests = res.wald_test_terms()
        # wtests is usually an OrderedDict-like mapping term names to results.
        # Find the key corresponding to Genus (term name will include 'Genus').
        omnibus_entry = None
        for key in wtests:
            if 'Genus' in str(key):
                omnibus_entry = wtests[key]
                break
        if omnibus_entry is not None:
            # omnibus_entry may have attributes .statistic and .pvalue (or .p_f and .pvalue)
            stat = None
            pval = None
            if hasattr(omnibus_entry, 'statistic'):
                stat = float(omnibus_entry.statistic)
            if hasattr(omnibus_entry, 'pvalue'):
                pval = float(omnibus_entry.pvalue)
            # Some versions return a Results object where .p_f exists:
            if pval is None and hasattr(omnibus_entry, 'p_f'):
                try:
                    pval = float(omnibus_entry.p_f)
                except Exception:
                    pval = None
            if stat is not None or pval is not None:
                genus_results['_genus_omnibus'] = {'statistic': stat, 'p': pval}
    except Exception:
        # If omnibus test not available, skip silently.
        pass

    description = (
        "For each non-reference genus (the model used treatment coding with Homo sapiens as the "
        "reference), the returned 'object' contains the log-odds coefficient (coef), its standard "
        "error (se), Wald p-value (p), 95% CI on the log-odds scale (ci_lower, ci_upper), and the "
        "exponentiated coefficient as an odds ratio with its CI (odds_ratio, odds_ratio_ci). "
        "A positive coef (odds_ratio > 1) means that genus has higher odds of AMTL compared to "
        "Homo sapiens; a negative coef (odds_ratio < 1) means lower odds. If present, an omnibus "
        "Wald test for the Genus term is provided under '_genus_omnibus' (statistic and p)."
    )

    return {'object': genus_results, 'description': description}