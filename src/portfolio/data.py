import numpy as np
import pandas as pd
import requests

# Hard-coded 200-stock universe
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
    # Materials (10)
    'LIN',  'APD',  'ECL',  'SHW',   'NEM',  'FCX',  'NUE',  'VMC',
    'MLM',  'PKG',
    # Utilities (10)
    'NEE',  'DUK',  'SO',   'D',     'AEP',  'EXC',  'XEL',  'SRE',
    'ED',   'ETR',
    # Real Estate (5)
    'AMT',  'PLD',  'CCI',  'EQIX',  'PSA'
]

def generate_synthetic_data(n_stocks: int = 179, seed: int = 42):
    """
    Generate synthetic stock returns using an 8-sector factor model mimicking S&P 500.
    Train: 11 years (2012-2022, 2767 days) | Test: 2 years (2023-2025, 501 days)
    """
    np.random.seed(seed)
    
    n_train = 2767
    n_test = 501
    total = n_train + n_test
    
    sector_cfg = [
        ('TECH', 0.0005, 0.012),
        ('HLT',  0.0003, 0.008),
        ('FIN',  0.0003, 0.009),
        ('ENE',  0.0002, 0.011),
        ('CDIS', 0.0004, 0.010),
        ('CST',  0.0002, 0.007),
        ('IND',  0.0003, 0.009),
        ('UTIL', 0.0002, 0.008),
    ]
    
    n_sec = len(sector_cfg)
    per_sector = n_stocks // n_sec
    market_ret = np.random.normal(0.0004, 0.009, total)
    
    all_ret, tickers = [], []
    
    for s_idx, (label, drift, vol) in enumerate(sector_cfg):
        sec_ret = np.random.normal(drift, vol * 0.5, total)
        n_in_sec = per_sector if s_idx < n_sec - 1 else n_stocks - per_sector * (n_sec - 1)
        
        for k in range(n_in_sec):
            beta = np.random.uniform(0.7, 1.3)
            idio = np.random.normal(0, np.random.uniform(0.006, 0.015), total)
            r = beta * market_ret + sec_ret + idio
            all_ret.append(r)
            tickers.append(f"{label}{k+1:02d}")
            
    arr = np.array(all_ret).T
    dates = pd.date_range('2012-01-01', periods=total, freq='B')
    df = pd.DataFrame(arr, index=dates, columns=tickers)
    
    train_ret = df.iloc[:n_train]
    test_ret = df.iloc[n_train:]
    
    tickers = list(df.columns)
    mu = train_ret.mean().values # Daily mean return
    cov = train_ret.cov().values # Daily cov matrix
    
    return tickers, mu, cov, train_ret, test_ret

def load_data(n_stocks: int = 179, seed: int = 42):
    """
    Load dataset for portfolio optimization.
    """
    tickers, mu, cov, train_ret, test_ret = generate_synthetic_data(n_stocks, seed)
    source = "S&P 500 Sector Factor Model (179 assets, 2012–2025)"
    print(f"    Loaded dataset [{source}]. Train={len(train_ret)} days, Test={len(test_ret)} days.")
    return tickers, mu, cov, train_ret, test_ret, source
