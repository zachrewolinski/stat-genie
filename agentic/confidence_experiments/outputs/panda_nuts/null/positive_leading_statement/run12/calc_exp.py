import math
coef = {
    'poisson_sex_m': -0.6750579610639305,
    'poisson_age': -0.032905308302552345,
    'poisson_help_y': -0.08712886445988799,
    'nb_sex_m': -0.701779405537351,
    'nb_age': 0.032729272554396394,
    'nb_help_y': -0.6184459322799958,
}
print({k: math.exp(v) for k,v in coef.items()})
