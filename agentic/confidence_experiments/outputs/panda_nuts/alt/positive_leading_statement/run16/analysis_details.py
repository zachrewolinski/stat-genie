import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("panda_nuts.csv")
formula = "nuts_opened ~ age + C(sex) + C(help)"
offset = np.log(df["seconds"])
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=offset,
)
res = model.fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

params = res.params
se = res.bse
pvals = res.pvalues
irr = np.exp(params)
ci_low = np.exp(params - 1.96 * se)
ci_high = np.exp(params + 1.96 * se)

summary = pd.DataFrame(
    {
        "coef": params,
        "se": se,
        "p": pvals,
        "IRR": irr,
        "IRR_CI_low": ci_low,
        "IRR_CI_high": ci_high,
    }
)
print(summary)
