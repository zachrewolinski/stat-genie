import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('mortgage.csv')

binary_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0,1})]

# Use all other columns as predictors (drop ID-like 'bad_history' maybe?).
# Identify ID-like column by being nearly unique.
unique_counts = {c: df[c].nunique() for c in df.columns}

# consider columns with near number of rows as id-like
id_like = [c for c,u in unique_counts.items() if u > 0.9*len(df)]

print('id_like', id_like)

results = {}

for ycol in binary_cols:
    X = df.drop(columns=[ycol] + id_like)
    y = df[ycol]
    data = pd.concat([X,y], axis=1).dropna()
    X = data.drop(columns=[ycol])
    y = data[ycol]

    # one-hot? all numeric already.
    # simple logistic with L2
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:,1]
        aucs.append(roc_auc_score(y_test, probs))
    results[ycol] = np.mean(aucs)

print('AUCs')
for k,v in sorted(results.items(), key=lambda x: -x[1]):
    print(k, v)
