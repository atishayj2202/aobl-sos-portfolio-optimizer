import os
import numpy as np
import pandas as pd
import yfinance as yf

# 179 S&P 500 Tickers
SP500_TICKERS = [
    # Technology (40)
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'TSLA', 'CRM',
    'ORCL', 'AMD',  'INTU', 'QCOM',  'TXN',  'ADI',  'AMAT', 'LRCX',
    'KLAC', 'MCHP', 'MU',   'CDNS',  'SNPS', 'ANSS', 'CTSH', 'ACN',
    'IBM',  'HPQ',  'DELL', 'STX',   'WDC',  'NTAP', 'KEYS', 'TRMB',
    'FFIV', 'JNPR', 'CIEN', 'AKAM',  'VRT',  'GDDY', 'GEN',  'EPAM',
    # Healthcare (35)
    'UNH',  'JNJ',  'LLY',  'ABBV',  'MRK',  'TMO',  'ABT',  'DHR',
    'BMY',  'AMGN', 'ISRG', 'REGN',  'VRTX', 'GILD', 'CI',   'HUM',
    'CVS',  'MCK',  'SYK',  'MDT',   'BDX',  'EW',   'ZTS',  'IDXX',
    'BSX',  'BAX',  'IQV',  'A',     'DXCM', 'HOLX', 'MTD',  'WAT',
    'MRNA', 'BIIB', 'ILMN',
    # Financials (30)
    'JPM',  'BAC',  'WFC',  'GS',    'MS',   'BLK',  'AXP',  'SPGI',
    'MCO',  'USB',  'TFC',  'PNC',   'COF',  'AIG',  'MET',  'PRU',
    'AFL',  'ALL',  'CB',   'TRV',   'MMC',  'AON',  'ICE',  'CME',
    'NDAQ', 'CBOE', 'SCHW', 'RJF',   'STT',  'BK',
    # Consumer Discretionary (20)
    'AMZN', 'HD',   'MCD',  'NKE',   'SBUX', 'TGT',  'LOW',  'BKNG',
    'MAR',  'HLT',  'YUM',  'DRI',   'ROST', 'TJX',  'LVS',  'MGM',
    'PHM',  'LEN',  'DHI',  'NVR',
    # Consumer Staples (15)
    'PG',   'KO',   'PEP',  'MDLZ',  'PM',   'MO',   'CL',   'EL',
    'KMB',  'GIS',  'HSY',  'K',     'SJM',  'CAG',  'CPB',
    # Energy (15)
    'XOM',  'CVX',  'COP',  'SLB',   'EOG',  'PXD',  'MPC',  'PSX',
    'VLO',  'HAL',  'BKR',  'DVN',   'FANG', 'OXY',  'HES',
    # Industrials (20)
    'RTX',  'HON',  'CAT',  'DE',    'UPS',  'FDX',  'LMT',  'GE',
    'NOC',  'BA',   'GD',   'MMM',   'EMR',  'ETN',  'ITW',  'PH',
    'ROK',  'CMI',  'IR',   'XYL',
    # Materials (4)
    'LIN',  'APD',  'ECL',  'SHW'
]

def generate_adjusted_synthetic_data(n_stocks: int = 179, seed: int = 42):
    """
    Generate synthetic stock returns mimicking S&P 500.
    Adjusted to have a realistic ~10% annual market drift.
    """
    np.random.seed(seed)
    
    n_train = 2767
    n_test = 501
    total = n_train + n_test
    
    sector_cfg = [
        ('TECH', 0.0004, 0.012),
        ('HLT',  0.0003, 0.008),
        ('FIN',  0.0003, 0.009),
        ('ENE',  0.0002, 0.011),
        ('CDIS', 0.0003, 0.010),
        ('CST',  0.0002, 0.007),
        ('IND',  0.0003, 0.009),
        ('UTIL', 0.0002, 0.008),
    ]
    
    n_sec = len(sector_cfg)
    per_sector = n_stocks // n_sec
    # Baseline market drift adjusted to realistic 10% annual (0.0004 daily mean)
    market_ret = np.random.normal(0.0004, 0.01, total)
    
    all_ret, tickers = [], []
    
    for s_idx, (label, drift, vol) in enumerate(sector_cfg):
        sec_ret = np.random.normal(drift, vol * 0.5, total)
        n_in_sec = per_sector if s_idx < n_sec - 1 else n_stocks - per_sector * (n_sec - 1)
        
        for k in range(n_in_sec):
            beta = np.random.uniform(0.7, 1.3)
            idio = np.random.normal(0, np.random.uniform(0.006, 0.015), total)
            # Make the return realistic. Average should be ~ 8-12% annually.
            r = beta * market_ret + sec_ret + idio
            all_ret.append(r)
            tickers.append(f"{label}{k+1:02d}")
            
    arr = np.array(all_ret).T
    dates = pd.date_range('2012-01-01', periods=total, freq='B')
    df = pd.DataFrame(arr, index=dates, columns=tickers)
    
    train_ret = df.iloc[:n_train]
    test_ret = df.iloc[n_train:]
    
    tickers = list(df.columns)
    mu = train_ret.mean().values
    cov = train_ret.cov().values
    
    return tickers, mu, cov, train_ret, test_ret

def load_real_data(n_stocks: int = 179):
    """Attempt to load real S&P 500 data from yfinance. Fall back to synthetic."""
    cache_file = "src/portfolio/sp500_daily_v2.csv"
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Drop columns with NaNs to ensure clean data
        df = df.dropna(axis=1)
        # Take up to n_stocks
        cols = df.columns[:n_stocks]
        df = df[cols]
        print(f"Loaded real empirical data from cache ({len(cols)} assets).")
    else:
        try:
            print("Attempting to fetch real data from yfinance...")
            data = yf.download(SP500_TICKERS[:n_stocks], start="2012-01-01", end="2025-01-01")['Adj Close']
            # Calculate daily returns
            df = data.pct_change().dropna(how='all')
            df = df.dropna(axis=1) # Drop assets that don't have full history
            df.to_csv(cache_file)
            print(f"Successfully downloaded and cached real data ({len(df.columns)} assets).")
        except Exception as e:
            print(f"Failed to fetch real data: {e}")
            print("Falling back to realistically scaled synthetic data...")
            return generate_adjusted_synthetic_data(n_stocks=n_stocks, seed=42)

    n_train = int(len(df) * 0.8) # 80/20 split
    train_ret = df.iloc[:n_train]
    test_ret = df.iloc[n_train:]
    
    tickers = list(df.columns)
    mu = train_ret.mean().values
    cov = train_ret.cov().values
    
    return tickers, mu, cov, train_ret, test_ret, "Real Empirical S&P 500 Data (2012-2025)"

def load_data_v2(n_stocks: int = 179, seed: int = 42):
    """Main entrypoint for v2 data loader."""
    tickers, mu, cov, train_ret, test_ret = generate_adjusted_synthetic_data(n_stocks, seed)
    source = "Adjusted Synthetic Factor Model (179 assets, 10% target drift)"
    
    print(f"    Loaded dataset [{source}]. Train={len(train_ret)} days, Test={len(test_ret)} days.")
    return tickers, mu, cov, train_ret, test_ret, source

if __name__ == "__main__":
    load_data_v2()
