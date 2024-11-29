import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from pykrx import stock
import statsmodels.api as sm
import math
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from tqdm import tqdm
import random


volume_top_6_tickers = ['005930', '000660', '015760', '105560', '055550', '017670']
volume_bottom_6_tickers = ['051900', '000810', '034730', '018260', '011170', '090430']
cap_top_6_tickers = ['005930', '000660', '005935', '005380', '207940', '005490']
cap_bottom_6_tickers = ['251270', '011170', '000810', '000270', '010950', '086790']
beta_top_6_tickers = ['096770', '005490', '051910', '017670', '066570', '035720']
beta_bottom_6_tickers = ['207940', '018260', '000810', '017670', '105560', '055550']
volatility_top_6_tickers = ['207940', '251270', '090430', '006400', '018260', '011170']
volatility_bottom_6_tickers = ['033780', '000810', '032830', '055550', '017670', '034730']

### 추가
df = pd.read_csv('kospi30.csv')

# 종목코드 6자리 맞추기 (6자리가 아니면 앞에 0을 추가)
df['종목코드'] = df['종목코드'].apply(lambda x: str(x).zfill(6))

# 종목코드만 뽑아서 리스트에 넣기
tickers = df['종목코드'].tolist()
###


# 일일 수익률 계산 함수
def calculate_daily_returns(ticker, start_date, end_date):
    # 주식의 일별 종가 가져오기
    df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)

    if df.empty:
        return None

    # 일별 수익률 계산 (로그 수익률)
    df['Log Return'] = np.log(df['종가'] / df['종가'].shift(1))

    # 수익률만 추출
    return df['Log Return']


class FactorMDP:
    def __init__(self, factor_returns, factor_loadings, asset_cov):
        """
        Initialize the FactorMDP class.
        :param factor_returns: pd.DataFrame - Factor returns (columns: factors, rows: time periods).
        :param factor_loadings: pd.DataFrame - Factor loadings for assets (rows: assets, columns: factors).
        :param residual_cov: np.array - Residual covariance matrix (diagonal matrix for idiosyncratic risk).
        """
        self.factor_returns = factor_returns
        self.factor_loadings = factor_loadings
        self.asset_cov = asset_cov
        # Compute factor covariance matrix
        self.factor_cov = np.cov(factor_returns, rowvar=False)
        
    def grad_x(self, x, z, u, rho):
        """
        Gradient of x with respect to the objective function.
        """
        return self.asset_cov @ x + rho * (x - z + u)

    def grad_z(self, x, z, u, rho):
        """
        Gradient of z with respect to the objective function.
        """
        return rho * (z - x - u)

    def update_u(self, u, x, z):
        """
        Update the dual variable u.
        """
        return u + (x - z)

    def gradient_descent(self, grad_func, var, args, step_size, max_iter=500):
        """
        Perform gradient descent to optimize a variable.
        """
        for _ in range(max_iter):
            grad = grad_func(var, *args)
            var -= step_size * grad
            var = np.maximum(var, 0)  # Ensure non-negativity
        return var

    def solve(self, rho=0.1, alpha=0.01, max_iter=1000, verbose=False):
        """
        Solve the MDP using ADMM.
        :param rho: float - Penalty parameter for ADMM.
        :param alpha: float - Learning rate for gradient descent.
        :param max_iter: int - Number of iterations for ADMM.
        :param verbose: bool - Print intermediate results.
        """
        # Initialize variables
        n_assets = self.asset_cov.shape[0]
        x = np.random.rand(n_assets)
        x /= x.sum()  # Normalize to sum to 1
        z = np.random.rand(n_assets)
        z /= z.sum()  # Normalize to sum to 1
        u = np.zeros_like(x)

        for i in range(max_iter):
            # Update x using gradient descent
            x = self.gradient_descent(self.grad_x, x, (z, u, rho), alpha)

            # Update z with projection onto constraints
            z = x + u
            z = np.maximum(z, 0)  # Ensure non-negativity
            z /= z.sum()  # Ensure sum to 1

            # Update u
            u = self.update_u(u, x, z)

            if verbose and i % 100 == 0:
                obj = x.T @ self.asset_cov @ x
                print(f"Iteration {i}, Objective: {obj:.6f}")

        return x

class FactorRB:
    def __init__(self, asset_rets, factor_exposure, factor_weights):
        """
        초기화 메서드: 자산 수익률, 팩터 노출도, 팩터 가중치를 설정합니다.
        """
        self.asset_rets = asset_rets.apply(pd.to_numeric, errors='coerce').dropna()
        self.factor_exposure = factor_exposure.apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
        self.factor_exposure = np.delete(self.factor_exposure, 0, axis=1)  # 티커 제거
        self.factor_weights = factor_weights
        self.num_assets = self.asset_rets.shape[1]

        # 잔차 공분산 행렬 (Idiosyncratic Risk)
        self.idiosyncratic_cov = np.diag(self.asset_rets.var())

        # 팩터 공분산 행렬 (Factor Risk)
        self.factor_cov = np.cov(self.factor_exposure, rowvar=False)

        # 초기 가중치: 동등 가중치
        self.w0 = np.ones(self.num_assets) / self.num_assets

    def obj_fun(self, x, factor_cov, idiosyncratic_cov, factor_exposure, rb):

        weighted_factor_exposure = factor_exposure * self.factor_weights[np.newaxis, :]
        factor_risk_contrib = x * np.dot(weighted_factor_exposure @ factor_cov @ weighted_factor_exposure.T, x)
        idiosyncratic_risk_contrib = x * np.dot(idiosyncratic_cov, x)
        total_risk = np.sum(factor_risk_contrib + idiosyncratic_risk_contrib)
        risk_contributions = (factor_risk_contrib + idiosyncratic_risk_contrib) / total_risk
        return np.sum((risk_contributions - rb) ** 2)

    def cons_sum_weight(self, x):
        """
        제약 조건: 가중치 합계는 1이 되어야 합니다.
        """
        return np.sum(x) - 1.0

    def cons_long_only_weight(self, x):
        """
        제약 조건: 모든 가중치는 0 이상이어야 합니다. (Long-only 조건)
        """
        return x

    def optimize(self, risk_budget):
        """
        팩터 기반 Risk Budgeting 최적화 함수.
        """
        cons = (
            {'type': 'eq', 'fun': self.cons_sum_weight},  # 가중치 합계 = 1
            {'type': 'ineq', 'fun': self.cons_long_only_weight}  # Long-only (가중치 >= 0)
        )

        # 최적화 수행 (SLSQP)
        result = minimize(
            self.obj_fun,
            self.w0,
            args=(self.factor_cov, self.idiosyncratic_cov, self.factor_exposure, risk_budget),
            method='SLSQP',
            constraints=cons,
            options={'disp': True}  # 최적화 과정 출력
        )
        return result.x

vkospi = pd.read_csv('KOSPI_Volatility.csv')
vkospi['날짜'] = pd.to_datetime(vkospi['날짜'])
vkospi['변동 %'] = vkospi['변동 %'].str.replace('%', '', regex=False).astype(float)

kospi = pd.read_csv('Kospi.csv',encoding='cp949')


date = kospi['일자']
    
#Kospi 125일 이동평균선 스프레드 = 모멘텀
kospi['spread'] = kospi['종가'] - kospi['종가'].rolling(window=125, min_periods=1).mean()

#Vkospi 50일 이동평균선 스프레드
vkospi['spread'] = vkospi['종가'].rolling(window=50, min_periods=1).mean() - vkospi['종가']

#Market Weather
sp_1 = kospi['spread'] 
sp_2 = vkospi['spread']

#min-Max 스케일링
sp_1_min = sp_1.min()
sp_1_Max = sp_1.max()

sp_2_min = sp_2.min()
sp_2_Max = sp_2.max()

sp_1_Scaled = (sp_1 - sp_1_min) / (sp_1_Max - sp_1_min)
sp_2_Scaled = (sp_2 - sp_2_min) / (sp_2_Max - sp_2_min)

MW = (sp_1_Scaled + sp_2_Scaled)*50

MW = MW.tolist()

MW = dict(zip(date, MW))

#전날 종가 기준으로 다음날 리밸런싱 시그널 발생
#시그널 0, 1, -1(홀드, 불, 베어)
def rebalance_signal(date, signal):

    if (signal[0] == -1) and MW[date] > 65:
            #70점 이상이면 리밸런싱
            signal[1] = 1
            signal[0] = 1

    elif (signal[0] == 1) and MW[date] < 50:
        signal[1] = -1
        signal[0] = -1

    else:
        signal[1] = 0

        
    return signal

# 종가 불러오기
def portfolio_price(date):

    portfolio_price_dict = {}

    date = date.replace('-','')

    # 날짜를 datetime 객체로 변환
    start_date = datetime.strptime(date, "%Y%m%d")
        
    # 다음날 계산
    end_date = start_date + timedelta(days=1)
       
    # 다시 문자열 형태로 변환 (YYYYMMDD)
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")


    for i in tickers:
        # 주식의 일별 종가 가져오기

        df = stock.get_market_ohlcv_by_date(start_date_str, end_date_str, i)

        if df.empty:
            close_price = 10000
            print("ErrorErrorErrorErrorErrorErrorErrorErrorErrorErrorErrorErrorErrorErrorErrorError")

        else:
            close_price = df['종가'].iloc[0]


        portfolio_price_dict[i] = close_price

    return portfolio_price_dict

def MDP(date):
    date = date.replace('-','')

    # 오늘 날짜 설정
    end_date = date

    # 입력 날짜를 datetime 형식으로 변환
    date = datetime.strptime(date, "%Y%m%d")
    
    # 1년 전 날짜 계산
    one_year_ago = date.replace(year=date.year - 1)
    
    start_date = one_year_ago.strftime("%Y%m%d")

    returns_volume_top = {}
    for ticker in volume_top_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_volume_top[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")


    returns_volume_bottom = {}
    for ticker in volume_bottom_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_volume_bottom[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")

    # volume_top_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    volume_top_6_daily_returns = pd.DataFrame(returns_volume_top)

    # volume_bottom_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    volume_bottom_6_daily_returns = pd.DataFrame(returns_volume_bottom)


    returns_beta_top = {}
    for ticker in beta_top_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_beta_top[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")


    returns_beta_bottom = {}
    for ticker in beta_bottom_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_beta_bottom[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")

    # volume_top_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    beta_top_6_daily_returns = pd.DataFrame(returns_beta_top)

    # volume_bottom_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    beta_bottom_6_daily_returns = pd.DataFrame(returns_beta_bottom)

    returns_volatility_top = {}
    for ticker in volatility_top_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_volatility_top[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")


    returns_volatility_bottom = {}
    for ticker in volatility_bottom_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_volatility_bottom[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")

    # volume_top_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    volatility_top_6_daily_returns = pd.DataFrame(returns_volatility_top)

    # volume_bottom_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    volatility_bottom_6_daily_returns = pd.DataFrame(returns_volatility_bottom)


    # Volume Top 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    volume_top_6_daily_returns['Average Log Return'] = volume_top_6_daily_returns.mean(axis=1)

    # Volume Bottom 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    volume_bottom_6_daily_returns['Average Log Return'] = volume_bottom_6_daily_returns.mean(axis=1)

    # Beta Top 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    beta_top_6_daily_returns['Average Log Return'] = beta_top_6_daily_returns.mean(axis=1)

    # Beta Bottom 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    beta_bottom_6_daily_returns['Average Log Return'] = beta_bottom_6_daily_returns.mean(axis=1)

    # Volatility Top 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    volatility_top_6_daily_returns['Average Log Return'] = volatility_top_6_daily_returns.mean(axis=1)

    # Volatility Bottom 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    volatility_bottom_6_daily_returns['Average Log Return'] = volatility_bottom_6_daily_returns.mean(axis=1)


    # 1. Volume 팩터 (Top 6 vs Bottom 6) 의 평균 수익률 차이 계산
    volume_avg_return_diff = volume_top_6_daily_returns['Average Log Return'] - volume_bottom_6_daily_returns['Average Log Return']

    # 3. Beta 팩터 (Top 6 vs Bottom 6) 의 평균 수익률 차이 계산
    beta_avg_return_diff = beta_top_6_daily_returns['Average Log Return'] - beta_bottom_6_daily_returns['Average Log Return']

    # 4. Volatility 팩터 (Top 6 vs Bottom 6) 의 평균 수익률 차이 계산
    volatility_avg_return_diff = volatility_top_6_daily_returns['Average Log Return'] - volatility_bottom_6_daily_returns['Average Log Return']



    factors_return = pd.DataFrame({
        'Volume': volume_avg_return_diff,
        'Beta': beta_avg_return_diff,
        'Volatility': volatility_avg_return_diff
    })


    factor_exposure = pd.DataFrame()
    assets_residuals = pd.DataFrame()

    for i in tickers:
        daily_returns = calculate_daily_returns(i, start_date, end_date)

        # daily_returns와 factors_return을 하나의 DataFrame으로 결합
        df = pd.concat([daily_returns, factors_return], axis=1)

        df = df.dropna()  # 결측치 제거

        # 독립변수와 종속변수 설정
        X = df[['Volume', 'Beta', 'Volatility']]  # 독립변수
        y = df['Log Return']  # 종속변수

        # X에 상수항 추가 회귀식에서 절편을 포함시키기 위해
        X = sm.add_constant(X)

        # 회귀 모델을 적합 회귀 분석
        model = sm.OLS(y, X)  # OLS Ordinary Least Squares 회귀 모델
        results = model.fit()  # 모델 적합
        coefficient = pd.DataFrame([results.params])
        coefficient.insert(0, 'Asset', i)
        factor_exposure = pd.concat([factor_exposure, coefficient], axis = 0)

        residuals = list(results.resid)
        assets_residuals[i] = residuals

        # 회귀 결과 출력

    factor_exposure = factor_exposure.drop('const', axis = 1)
    assets_cov = assets_residuals.cov()

    if __name__ == "__main__":
        factor_returns = factors_return
        factor_loadings = factor_exposure
        asset_cov = assets_cov
        factor_mdp = FactorMDP(factor_returns, factor_loadings, asset_cov)
        optimal_weights = factor_mdp.solve(rho=0.1, alpha=0.01, max_iter=20, verbose=True)
        weights = optimal_weights.to_dict()
    return weights

def RB(date):
    date = date.replace('-', '')

    # 오늘 날짜 설정
    end_date = date

    # 입력 날짜를 datetime 형식으로 변환
    date = datetime.strptime(date, "%Y%m%d")

    # 1년 전 날짜 계산
    one_year_ago = date.replace(year=date.year - 1)

    start_date = one_year_ago.strftime("%Y%m%d")

    returns_volume_top = {}
    for ticker in volume_top_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_volume_top[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")

    returns_volume_bottom = {}
    for ticker in volume_bottom_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_volume_bottom[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")

    # volume_top_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    volume_top_6_daily_returns = pd.DataFrame(returns_volume_top)

    # volume_bottom_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    volume_bottom_6_daily_returns = pd.DataFrame(returns_volume_bottom)

    returns_beta_top = {}
    for ticker in beta_top_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_beta_top[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")

    returns_beta_bottom = {}
    for ticker in beta_bottom_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_beta_bottom[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")

    # volume_top_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    beta_top_6_daily_returns = pd.DataFrame(returns_beta_top)

    # volume_bottom_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    beta_bottom_6_daily_returns = pd.DataFrame(returns_beta_bottom)

    returns_volatility_top = {}
    for ticker in volatility_top_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_volatility_top[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")

    returns_volatility_bottom = {}
    for ticker in volatility_bottom_6_tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        if daily_returns is not None:
            returns_volatility_bottom[ticker] = daily_returns
        else:
            print(f"Data not available for {ticker}")

    # volume_top_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    volatility_top_6_daily_returns = pd.DataFrame(returns_volatility_top)

    # volume_bottom_6_daily_returns 데이터프레임 생성 (날짜는 인덱스, 종목은 컬럼)
    volatility_bottom_6_daily_returns = pd.DataFrame(returns_volatility_bottom)

    # Volume Top 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    volume_top_6_daily_returns['Average Log Return'] = volume_top_6_daily_returns.mean(axis=1)

    # Volume Bottom 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    volume_bottom_6_daily_returns['Average Log Return'] = volume_bottom_6_daily_returns.mean(axis=1)

    # Beta Top 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    beta_top_6_daily_returns['Average Log Return'] = beta_top_6_daily_returns.mean(axis=1)

    # Beta Bottom 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    beta_bottom_6_daily_returns['Average Log Return'] = beta_bottom_6_daily_returns.mean(axis=1)

    # Volatility Top 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    volatility_top_6_daily_returns['Average Log Return'] = volatility_top_6_daily_returns.mean(axis=1)

    # Volatility Bottom 6 종목 일일 수익률에 대해 행별 평균 계산하여 새로운 열 추가
    volatility_bottom_6_daily_returns['Average Log Return'] = volatility_bottom_6_daily_returns.mean(axis=1)

    # 1. Volume 팩터 (Top 6 vs Bottom 6) 의 평균 수익률 차이 계산
    volume_avg_return_diff = volume_top_6_daily_returns['Average Log Return'] - volume_bottom_6_daily_returns[
        'Average Log Return']

    # 3. Beta 팩터 (Top 6 vs Bottom 6) 의 평균 수익률 차이 계산
    beta_avg_return_diff = beta_top_6_daily_returns['Average Log Return'] - beta_bottom_6_daily_returns[
        'Average Log Return']

    # 4. Volatility 팩터 (Top 6 vs Bottom 6) 의 평균 수익률 차이 계산
    volatility_avg_return_diff = volatility_top_6_daily_returns['Average Log Return'] - \
                                 volatility_bottom_6_daily_returns['Average Log Return']

    factors_return = pd.DataFrame({
        'Volume': volume_avg_return_diff,
        'Beta': beta_avg_return_diff,
        'Volatility': volatility_avg_return_diff
    })

    factor_exposure = pd.DataFrame()
    assets_residuals = pd.DataFrame()

    for i in tickers:
        daily_returns = calculate_daily_returns(i, start_date, end_date)

        # daily_returns와 factors_return을 하나의 DataFrame으로 결합
        df = pd.concat([daily_returns, factors_return], axis=1)

        df = df.dropna()  # 결측치 제거

        # 독립변수와 종속변수 설정
        X = df[['Volume', 'Beta', 'Volatility']]  # 독립변수
        y = df['Log Return']  # 종속변수

        # X에 상수항 추가 회귀식에서 절편을 포함시키기 위해
        X = sm.add_constant(X)

        # 회귀 모델을 적합 회귀 분석
        model = sm.OLS(y, X)  # OLS Ordinary Least Squares 회귀 모델
        results = model.fit()  # 모델 적합
        coefficient = pd.DataFrame([results.params])
        coefficient.insert(0, 'Asset', i)
        factor_exposure = pd.concat([factor_exposure, coefficient], axis=0)

        residuals = list(results.resid)
        assets_residuals[i] = residuals

    factor_exposure = factor_exposure.drop('const', axis=1)
    assets_cov = assets_residuals.cov()

    # 자산별 일일 수익률을 저장할 빈 데이터프레임
    all_returns = pd.DataFrame()

    # 각 자산에 대해 일일 수익률 계산 후 데이터프레임에 추가
    for ticker in tickers:
        daily_returns = calculate_daily_returns(ticker, start_date, end_date)
        all_returns[ticker] = daily_returns  # 각 자산의 수익률을

    # 메인 코드 실행
    if __name__ == "__main__":
        asset_rets = all_returns.apply(pd.to_numeric, errors='coerce').dropna()
        factor_exposure = factor_exposure.apply(pd.to_numeric, errors='coerce').dropna()
        factor_weights = np.array([0.3, 0.5, 0.2])

        # Risk Budget 정의 (종목별 동일 기여)
        num_assets = asset_rets.shape[1]
        risk_budget = np.ones(num_assets) / num_assets

        # 최적화 클래스 생성 및 실행
        optimizer = FactorRB(asset_rets, factor_exposure, factor_weights)
        optimal_weights = optimizer.optimize(risk_budget)

        # 최적화 결과 출력
        

        # 결과를 데이터프레임으로 정리
        weights_df = pd.DataFrame({
            'Ticker': tickers,
            'Optimal Weight': optimal_weights
        })

        # 가중치를 내림차순으로 정렬
        weights_df_sorted = weights_df.sort_values(by='Optimal Weight', ascending=False)
        weights_dict = weights_df_sorted.set_index('Ticker')['Optimal Weight'].to_dict()


    return weights_dict



#각 주식 개수
def stocks(total_stock_value, price_dict, weight_dict):
    stocks = {}
    for key, value in weight_dict.items():
        stocks[key] = math.floor(total_stock_value * value / price_dict[key])

    return stocks



#backtest
n = 0
#시작 자산 = 1억(6000만원 주식, 4000만원 무위험 자산)
currency = []
start_currency = 40000000
#포트폴리오 가치
stock_value = []
#시작 포트폴리오 가치
start_value = 60000000
#포트폴리오 비중
portfolio_weight = {}
#각 주식 가격
stock_price = {}
#각 주식 개수
stocks_dict = {}
#오늘 주식 개수
current_stocks = {}
#현재 포트폴리오 비중
current_portfolio_weight = {}
#리밸런싱 날짜
rebalanced_date = []

signal = [1, 0]

for i in tickers:
    stocks_dict[i] = []

for i in tickers:
    stock_price[i] = []

for i in tickers:
    portfolio_weight[i] = [1/30]

#비교지수(long_only)
long_only = []
#개수
long_only_stocks = {}

#시작 주식 개수 계산
start_price = portfolio_price('20190101')
added_currency = 0
for key, value in stocks_dict.items():
    stocks_dict[key] = [math.floor(start_value * portfolio_weight[key][n] / start_price[key])]
    #남는 현금 계산
    floor_ = math.floor(start_value * portfolio_weight[key][n] / start_price[key])
    non_floor_ = start_value * portfolio_weight[key][n] / start_price[key]
    added_currency = added_currency + non_floor_ - floor_


#비교지수 시작 개수 계산
for key, value in stocks_dict.items():
    long_only_stocks[key] = math.floor(start_value * (1/30) / start_price[key])
   

#시작 주식 개수 선언
for key, value in stocks_dict.items():
    current_stocks[key] = value[0]

current_currency = start_currency

remain_currency = 0

for i in date:
    #추가되는 순현금
    added_currency = 0
    #당일 주식 종가 계산
    current_portfolio_price = portfolio_price(i)

    #당일 주식 비중(리밸런싱 안하면 그대로)
    for key, value in portfolio_weight.items():
        current_portfolio_weight[key] = value[n]

    #당일 포트폴리오 가치 계산
    current_stock_value = 0
    for key, value in current_portfolio_price.items():
        dict_ = stocks_dict[key]
        current_stock_value += dict_[n] * value

    #자투리 현금 재투자
    current_stock_value = current_stock_value + remain_currency
        
    #당일 주식 개수 계산
    current_stocks

    #리밸런싱해야하는 비중 계산
    if signal[1] == -1:

        print("MDP...")

        current_portfolio_weight = MDP(i)

        rebalanced_date.append(i)

        #비중 계산 후 살 수 있는 주식 개수 계산
        for key, value in current_portfolio_price.items():

            weight = current_portfolio_weight[key]
            #리밸런스 해야하는 주식 개수
            rebalanced_stocks = math.floor(weight * current_stock_value / value)
            #남은 금액은 현금으로
            added_currency = added_currency + weight * current_stock_value - rebalanced_stocks * value
            #주식 개수 업데이트
            current_stocks[key] = rebalanced_stocks
    
    elif signal[1] == 1:

        print("RB...")

        current_portfolio_weight = RB(i)

        rebalanced_date.append(i)

        #비중 계산 후 살 수 있는 주식 개수 계산
        for key, value in current_portfolio_price.items():

            weight = current_portfolio_weight[key]
            #리밸런스 해야하는 주식 개수
            rebalanced_stocks = math.floor(weight * current_stock_value / value)
            #남은 금액은 현금으로
            added_currency = added_currency + weight * current_stock_value - rebalanced_stocks * value
            #주식 개수 업데이트
            current_stocks[key] = rebalanced_stocks

    #오늘 주식 개수 전체 딕셔너리에 기록
    for key, value in stocks_dict.items():
        list_ = stocks_dict[key]
        list_.append(current_stocks[key])
        stocks_dict[key] = list_

    #오늘 추가된 현금 기록
    current_currency = current_currency + added_currency
    currency.append(current_currency)

    #오늘 주식 가격 전체 딕셔너리에 기록
    for key, value in stock_price.items():
        list_ = stock_price[key]
        list_.append(current_portfolio_price[key])
        stock_price[key] = list_

    #당일 포트폴리오 가치 계산
    current_stock_value = 0
    for key, value in current_portfolio_price.items():
        current_stock_value += current_stocks[key] * value

    
######################비교지수 가치 계산######################
    current_long_value = 0
    for key, value in current_portfolio_price.items():
        current_long_value += long_only_stocks[key] * value

    long_only.append(current_long_value)
#############################################################

    #오늘 주식 비중 기록
    for key, value in portfolio_weight.items():
        #기존 weight
        list_ = value
        list_.append(current_portfolio_weight[key])
        portfolio_weight[key] = list_

    #포트폴리오 가치 기록
    stock_value.append(current_stock_value)
    

    #리밸런싱 시그널 확인
    signal = rebalance_signal(i, signal)

    #현금이 4000만원보다 많으면 다음날 짜투리 다시 재투자
    if current_currency > 40000000:
        remain_currency = current_currency - 40000000
        current_currency = 40000000
    else:
        remain_currency = 0
    
    n = n + 1

    print(i)
MW = list(MW.values())
total_value =  [x + y for x, y in zip(stock_value, currency)]

# DataFrame 생성
data = pd.DataFrame({
    'Date': date,
    'Total_Value': total_value, 
    'Market_Weather':MW,
    'Index':long_only
})

# CSV 파일로 저장
data.to_csv('backtest.csv', index=False)

# DataFrame 생성
data = pd.DataFrame({
    'rebalanced_date': rebalanced_date
})

# CSV 파일로 저장
data.to_csv('rebalanced_date.csv', index=False)

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(12, 6))

# MW 플롯 (파란색 반투명 선)
ax1.plot(date, MW, color='blue', alpha=0.5, label='Market Weather')

ax1.set_ylabel('Market Weather', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# y축에 수직선 추가 (초록색 선)
ax1.axhline(y=60, color='green', linestyle='--', linewidth=1, alpha=0.7, label='MW Threshold (60)')
ax1.axhline(y=50, color='green', linestyle='--', linewidth=1, alpha=0.7, label='MW Threshold (50)')

# x축에 리밸런싱 날짜 점선 추가 (검은색 점선)
for r_date in rebalanced_date:
    if r_date in date:
        ax1.axvline(x=r_date, color='black', linestyle=':', linewidth=0.8)

# total_value 플롯 (빨간색 선)
ax2 = ax1.twinx()
ax2.plot(date, total_value, color='red', label='Total Value (Portfolio)')
ax2.set_ylabel('Total Value', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# x축 레이블 회전
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 그래프 타이틀 및 레전드
fig.suptitle('Back Test Result')
fig.tight_layout()
plt.show()


    