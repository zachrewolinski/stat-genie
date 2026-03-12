import statsmodels.api as sm

try:
    d = sm.datasets.mortgage.load_pandas().data
    print(d.head())
    print(d.mean())
except Exception as e:
    print('error', e)
