import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')
print('shape', _df.shape)
print('columns', _df.columns.tolist())
print('\nhead')
print(_df.head())

# Check binary columns value counts
for col in _df.columns:
    if _df[col].dropna().nunique() <= 5:
        print('\nvalue counts', col)
        print(_df[col].value_counts(dropna=False))

# Check accept/deny consistency if both exist
if 'accept' in _df.columns and 'deny' in _df.columns:
    print('\naccept+deny unique pairs')
    print(_df[['accept','deny']].drop_duplicates().sort_values(['accept','deny']))
    print('accept mean', _df['accept'].mean(), 'deny mean', _df['deny'].mean())
    print('accept+deny==1 share', ((_df['accept'] + _df['deny'])==1).mean())

# Gender effect: use female if exists
if 'female' in _df.columns:
    # Choose outcome: use deny if exists, else accept
    outcome = 'deny' if 'deny' in _df.columns else 'accept'
    print('\nOutcome:', outcome)
    print('female mean', _df['female'].mean())
    # Crosstab
    ct = pd.crosstab(_df['female'], _df[outcome], normalize='index')
    print('\nCrosstab female vs outcome (row proportions)')
    print(ct)

    # Simple difference in outcome rates
    rate_f = _df.loc[_df['female']==1, outcome].mean()
    rate_m = _df.loc[_df['female']==0, outcome].mean()
    print('outcome rate female', rate_f, 'male', rate_m, 'diff (female-male)', rate_f-rate_m)

    # Logistic regression with female only
    y = _df[outcome]
    X = sm.add_constant(_df['female'])
    model = sm.Logit(y, X, missing='drop').fit(disp=False)
    print('\nLogit outcome ~ female')
    print(model.summary())

    # Logistic regression with controls if possible: use numeric columns excluding outcome
    # Select candidate controls: numeric columns with >2 unique values or binary but not outcome/female
    candidates = [c for c in _df.columns if c not in [outcome, 'female']]
    # filter numeric
    cand_num = [c for c in candidates if pd.api.types.is_numeric_dtype(_df[c])]
    # drop columns with too many missing
    cand_num = [c for c in cand_num if _df[c].isna().mean() < 0.5]
    # remove columns that look like IDs with high unique ratio
    cand_num = [c for c in cand_num if _df[c].nunique(dropna=True) / len(_df) < 0.9]
    # To avoid perfect multicollinearity, keep columns with variance
    cand_num = [c for c in cand_num if _df[c].nunique(dropna=True) > 1]

    # Build design matrix
    X2 = _df[['female'] + cand_num].copy()
    X2 = sm.add_constant(X2)
    # Fit model
    try:
        model2 = sm.Logit(y, X2, missing='drop').fit(disp=False)
        print('\nLogit outcome ~ female + controls')
        print('controls', cand_num)
        print(model2.summary())
    except Exception as e:
        print('Error fitting controlled model', e)
