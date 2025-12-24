import mpmath as mp
from scipy.stats import t

# Compute t-CDF using the regularized incomplete beta function
def t_cdf(t, df):
    x = df / (df + t**2)
    I = mp.betainc(df/2, 0.5, 0, x, regularized=True)
    if t >= 0:
        return 1 - 0.5*I
    else:
        return 0.5*I
    
def calculate_p(t_list, df_list, sided='two'):
    two_p_vals = []
    one_p_vals = []
    for i in range(len(df_list)):
        t_value=t_list[i]
        df = df_list[i]
        # p = (2*(1 - t_cdf(abs(t_value), df)))

        cdf_val = t.cdf(t_value, df)
        one_p_vals.append(cdf_val)
        two_p_vals.append( 2 * min(cdf_val, 1-cdf_val))
            
    print('TWO: ', two_p_vals)
    print('\n ONE: ', one_p_vals)

t_list = [
-18.42,
-2.15,
-15.45,
-2.63,
-14.36,
-2.91,
12.80,
-6.39,
-23.03,
14.01,
39.97,
-2.69


]
df_list = [
49.00,
49.01,
54.19,
49.00,
49.00,
50.84,
51.07,
81.99,
63.31,
89.04,
52.14,
95.25

]

calculate_p(t_list, df_list, 'two')
