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