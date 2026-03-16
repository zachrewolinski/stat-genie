import statsmodels.discrete.count_model as cm
import statsmodels.discrete.discrete_model as dm

print("count_model:", [x for x in dir(cm) if "Negative" in x])
print("discrete_model:", [x for x in dir(dm) if "Negative" in x])
