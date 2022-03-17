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


def portfolio_return(weights, returns):
    return weights.T @ returns

def portfolio_volatility(weights, cov):
    return (weights.T @ cov @ weights)**0.5