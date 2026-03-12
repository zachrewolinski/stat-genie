import math
coef=-0.16630560195360886
se=0.15038447103911068
lo=coef-1.96*se
hi=coef+1.96*se
print(lo,hi, math.exp(lo), math.exp(hi))
coef=-0.17420263716333706
se=0.14992056986749816
lo=coef-1.96*se
hi=coef+1.96*se
print(lo,hi, math.exp(lo), math.exp(hi))
