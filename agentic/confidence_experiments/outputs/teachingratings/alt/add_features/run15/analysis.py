import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('teachingratings.csv')

    # Basic correlation
    corr = df['beauty'].corr(df['eval'])

    # Simple OLS
    simple_model = smf.ols('eval ~ beauty', data=df).fit()

    # Controls based on the original teaching ratings dataset
    controls = [
        'age',
        'C(gender)',
        'C(minority)',
        'C(native)',
        'C(tenure)',
        'C(division)',
        'C(credits)',
        'students',
        'allstudents'
    ]
    formula = 'eval ~ beauty + ' + ' + '.join(controls)
    full_model = smf.ols(formula, data=df).fit()

    def get_pvalue(result, term):
        if hasattr(result, 'pvalues') and isinstance(result.pvalues, pd.Series):
            return result.pvalues.get(term)
        # Robust covariance results sometimes return ndarray pvalues
        try:
            names = result.model.exog_names
            idx = names.index(term)
            return float(result.pvalues[idx])
        except Exception:
            return np.nan

    # Cluster-robust SE by professor id (prof)
    try:
        cluster_model = full_model.get_robustcov_results(cov_type='cluster', groups=df['prof'])
    except Exception:
        cluster_model = full_model

    # Standardized coefficient (beta) for beauty
    df_std = df.copy()
    df_std['beauty_z'] = (df_std['beauty'] - df_std['beauty'].mean()) / df_std['beauty'].std(ddof=0)
    df_std['eval_z'] = (df_std['eval'] - df_std['eval'].mean()) / df_std['eval'].std(ddof=0)

    std_formula = 'eval_z ~ beauty_z + ' + ' + '.join(controls)
    std_model = smf.ols(std_formula, data=df_std).fit()
    try:
        std_cluster = std_model.get_robustcov_results(cov_type='cluster', groups=df['prof'])
    except Exception:
        std_cluster = std_model

    # Collect results
    results = {
        'n': len(df),
        'corr_beauty_eval': corr,
        'simple_coef': simple_model.params['beauty'],
        'simple_p': simple_model.pvalues['beauty'],
        'full_coef': full_model.params['beauty'],
        'full_p': full_model.pvalues['beauty'],
        'cluster_p': get_pvalue(cluster_model, 'beauty'),
        'std_beta': std_model.params['beauty_z'],
        'std_p': get_pvalue(std_model, 'beauty_z'),
        'std_cluster_p': get_pvalue(std_cluster, 'beauty_z'),
    }

    # Effect interpretation: predicted change in eval for 1 SD increase in beauty
    beauty_sd = df['beauty'].std(ddof=0)
    results['beauty_sd'] = beauty_sd
    results['eval_sd'] = df['eval'].std(ddof=0)
    results['pred_eval_change_per_sd_beauty'] = full_model.params['beauty'] * beauty_sd

    for k, v in results.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
