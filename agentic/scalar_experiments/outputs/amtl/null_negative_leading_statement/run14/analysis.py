import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_PATH = 'amtl.csv'


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
    df = df[cols].dropna().copy()
    df = df[df['sockets'] > 0].copy()
    df['prop_amtl'] = df['num_amtl'] / df['sockets']
    return df


def fit_model(df: pd.DataFrame):
    # Reference genus is Homo sapiens so coefficients for non-human genera are relative to humans
    formula = 'prop_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)'
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets'])
    result = model.fit()
    return result


def adjusted_predictions(result, df: pd.DataFrame):
    mean_age = df['age'].mean()
    mean_prob_male = df['prob_male'].mean()
    tooth_props = df['tooth_class'].value_counts(normalize=True).to_dict()

    def pred_for_genus(genus: str) -> float:
        rows = []
        weights = []
        for tooth_class, weight in tooth_props.items():
            rows.append({
                'genus': genus,
                'age': mean_age,
                'prob_male': mean_prob_male,
                'tooth_class': tooth_class,
            })
            weights.append(weight)
        design = pd.DataFrame(rows)
        preds = result.predict(design)
        return float(np.average(preds, weights=weights))

    genera = df['genus'].unique().tolist()
    preds = {g: pred_for_genus(g) for g in genera}

    # Weighted average of non-human genera based on sample counts
    non_human = [g for g in genera if g != 'Homo sapiens']
    counts = df['genus'].value_counts().to_dict()
    weights = np.array([counts[g] for g in non_human], dtype=float)
    weights = weights / weights.sum()
    non_human_avg = float(np.average([preds[g] for g in non_human], weights=weights))

    return preds, non_human_avg


def bootstrap_diff(df: pd.DataFrame, n_boot: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(df)
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, n)
        sample = df.iloc[sample_idx].copy()
        try:
            res = fit_model(sample)
            preds, non_human_avg = adjusted_predictions(res, sample)
            diff = preds['Homo sapiens'] - non_human_avg
            diffs.append(diff)
        except Exception:
            continue
    diffs = np.array(diffs, dtype=float)
    return diffs


def main():
    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)

    result = fit_model(df)
    preds, non_human_avg = adjusted_predictions(result, df)
    diff = preds['Homo sapiens'] - non_human_avg

    diffs = bootstrap_diff(df, n_boot=200, seed=7)
    if diffs.size:
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    else:
        ci_low, ci_high = float('nan'), float('nan')

    output = {
        'n_rows': int(len(df)),
        'params': result.params.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'preds': preds,
        'non_human_avg': non_human_avg,
        'diff_human_minus_nonhuman': diff,
        'boot_ci_2.5': ci_low,
        'boot_ci_97.5': ci_high,
        'boot_n': int(diffs.size),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
