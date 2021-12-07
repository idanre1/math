import statsmodels.stats.diagnostic as sm
import statsmodels.api as smi
import pandas as pd

def white_test(data: pd.DataFrame, window: int = 21, pval=0.05):
    '''
    https://www.statisticshowto.com/white-test/
    The null hypothesis for White’s test is that the variances for the errors are equal (homoscedastic)
    The alternate hypothesis (the one you’re testing), is that the variances are not equal (heteroscedastic)
    The larger the tStats value the more heteroscedastic the distibution
    '''
    data['std1'] = data['price'].rolling(window).std()
    data.dropna(inplace= True)
    X = smi.tools.tools.add_constant(data['price'])
    results = smi.regression.linear_model.OLS(data['std1'], X).fit()
    resid = results.resid
    exog = results.model.exog

    tStat, pVal = (sm.het_white(resid, exog)[0], sm.het_white(resid, exog)[1])
    if pVal <= pval:
        p = 'White test outcome at %s signficance: heteroscedastic' % pval
    else:
        p = 'White test outcome at %s signficance: cannot approve heteroscedastic' % pval
    return (tStat, pVal, p)