import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv('crofoot.csv')

    # Relative group size: focal minus other
    df['rel_size'] = df['n_focal'] - df['n_other']
    # Relative contest location: positive if contest is closer to focal group's home range center
    df['rel_location'] = df['dist_other'] - df['dist_focal']

    X = df[['rel_size', 'rel_location']]
    X = sm.add_constant(X)
    y = df['win']

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    # Save a concise summary to a CSV for inspection if needed
    summary = pd.DataFrame({
        'coef': result.params,
        'std_err': result.bse,
        'z': result.tvalues,
        'p_value': result.pvalues,
    })
    summary.to_csv('model_summary.csv')

    print(result.summary())


if __name__ == '__main__':
    main()
