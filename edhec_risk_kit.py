import numpy as np
import pandas as pd

def drawdown(return_series: pd.Series, initial_wealth: int = 1000) -> pd.Series:
    
    prices = initial_wealth * (return_series + 1).cumprod()
    peaks = prices.cummax()
    drawdowns = (prices - peaks) / peaks 
    return pd.DataFrame(data={
        'wealth_index': prices,
        'previous_peak': peaks,
        'drawdown': drawdowns
    })


def get_ffme_returns(small_cap_col='Lo 10', large_cap_col='Hi 10') -> pd.DataFrame:
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', na_values=-99.99, index_col=0)
    returns = me_m[[small_cap_col, large_cap_col]]
    returns.columns = ['SmallCap', 'LargeCap']
    returns = returns / 100
    returns.index = pd.to_datetime(returns.index, format='%Y%m').to_period('M')
    return returns

def get_hfi_returns() -> pd.DataFrame:
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv', header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_returns() -> pd.DataFrame:
    ind_df = pd.read_csv('data/ind30_m_vw_rets.csv', header=0, index_col=0, parse_dates=True) / 100
    ind_df.index = pd.to_datetime(ind_df.index, format='%Y%m').to_period('M')
    ind_df.columns = ind_df.columns.str.strip()
    return ind_df

def get_ind_size() -> pd.DataFrame:
    ind_df = pd.read_csv('data/ind30_m_size.csv', header=0, index_col=0, parse_dates=True)
    ind_df.index = pd.to_datetime(ind_df.index, format='%Y%m').to_period('M')
    ind_df.columns = ind_df.columns.str.strip()
    return ind_df


def get_ind_nfirms() -> pd.DataFrame:
    ind_df = pd.read_csv('data/ind30_m_nfirms.csv', header=0, index_col=0, parse_dates=True)
    ind_df.index = pd.to_datetime(ind_df.index, format='%Y%m').to_period('M')
    ind_df.columns = ind_df.columns.str.strip()
    return ind_df

def get_total_market_index_returns() -> pd.DataFrame:
    ind_return_df = get_ind_returns()
    ind_size_df = get_ind_size()
    ind_nfirm_df = get_ind_nfirms()
    ind_market_cap_df = ind_size_df * ind_nfirm_df
    ind_market_cap_weighted_df = ind_market_cap_df.divide(ind_market_cap_df.sum(axis=1), axis=0)
    total_market_returns = (ind_market_cap_weighted_df * ind_return_df).sum(axis=1)
    return total_market_returns


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())

def annualize_return(return_series: pd.Series) -> float:
    compounded_growth = (1 + return_series).prod()
    n_period = len(return_series)
    return compounded_growth**(12 / n_period) - 1

def annualize_voltatility(return_series: pd.Series) -> float:
    return return_series.std() * np.sqrt(12)

def sharpe_ratio(return_series: pd.Series, risk_free_rate: float = 0.02) -> float:
    annualize_risk_free = (1 + risk_free_rate) ** (1 / 12) - 1
    excess_returns = return_series - annualize_risk_free
    annualized_excess_return = annualize_return(excess_returns)
    annualized_voltatility = annualize_voltatility(return_series)
    return annualized_excess_return / annualized_voltatility

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def portfolio_return(weights, returns):
    return weights.T @ returns

def portfolio_volatility(weights, cov):
    return (weights.T @ cov @ weights)**0.5


def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_return)
    ann_vol = r.aggregate(annualize_voltatility)
    ann_sr = r.aggregate(sharpe_ratio, risk_free_rate=riskfree_rate)
    dd = r.aggregate(lambda r: drawdown(r)['drawdown'].min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })
