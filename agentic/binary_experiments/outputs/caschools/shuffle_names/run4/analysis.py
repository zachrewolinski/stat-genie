import pandas as pd
import statsmodels.api as sm

def main():
    df = pd.read_csv('caschools.csv')

    # Map shuffled columns to their actual meanings
    # students: total enrollment, teachers: full-time equivalent teachers
    students = df['english']
    teachers = df['students']
    str_ratio = students / teachers

    # Academic performance: average of reading and math scores
    read_score = df['district']
    math_score = df['expenditure']
    avg_score = (read_score + math_score) / 2

    # Correlation and simple OLS regression
    corr = avg_score.corr(str_ratio)
    X = sm.add_constant(str_ratio)
    model = sm.OLS(avg_score, X).fit()

    print('Correlation between student-teacher ratio and avg score:', corr)
    print(model.summary())

if __name__ == '__main__':
    main()
