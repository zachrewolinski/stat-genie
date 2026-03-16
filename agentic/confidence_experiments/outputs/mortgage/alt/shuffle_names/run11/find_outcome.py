import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv('mortgage.csv')

cont_cols = ['mortgage_credit','housing_expense_ratio','Unnamed: 0']

binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]

for b in binary_cols:
    sub = df[[b]+cont_cols].dropna()
    y = sub[b].values
    X = sub[cont_cols].values
    if len(np.unique(y))<2:
        continue
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs=[]
    for train, test in skf.split(X, y):
        model = LogisticRegression(max_iter=1000)
        model.fit(X[train], y[train])
        probs = model.predict_proba(X[test])[:,1]
        aucs.append(roc_auc_score(y[test], probs))
    print(b, 'mean_auc', np.mean(aucs))
