import pandas as pd
import statsmodels.formula.api as smf

_df = pd.read_csv('affairs.csv')
_df['children'] = _df['children'].astype('category')
_df['gender'] = _df['gender'].astype('category')

model = smf.negativebinomial('affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=_df)
res = model.fit(disp=False)

coef = res.params.get('C(children)[T.yes]')
se = res.bse.get('C(children)[T.yes]')
p = res.pvalues.get('C(children)[T.yes]')
import numpy as np
irr = float(np.exp(coef))

with open('analysis_nb_results.txt','w') as f:
    f.write(res.summary().as_text())
    f.write('\n\nchildren coef=%0.4f se=%0.4f p=%0.4g irr=%0.4f\n' % (coef,se,p,irr))
