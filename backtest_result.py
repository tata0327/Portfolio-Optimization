import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import numpy as np

df = pd.read_csv('backtest.csv')

date = df['Date']

# 문자열 날짜를 datetime 형식으로 변환
date = pd.to_datetime(date)

backtest = df['Total_Value']

long_only = df['Long_only']

mw = df['Market_Weather']

rebalanced = pd.read_csv('rebalanced_date.csv')

rebalanced_date = rebalanced['rebalanced_date']

# 문자열 날짜를 datetime 형식으로 변환
rebalanced_date = pd.to_datetime(rebalanced_date)

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(12, 6))

# MW 플롯 (파란색 반투명 선)
ax1.plot(date, mw, color='blue', alpha=0.5, label='Market Weather')

ax1.set_ylabel('Market Weather', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')


# y축에 수직선 추가 (초록색 선)
ax1.axhline(y=60, color='green', linestyle='--', linewidth=1, alpha=0.7, label='MW Threshold (60)')
ax1.axhline(y=50, color='green', linestyle='--', linewidth=1, alpha=0.7, label='MW Threshold (50)')

# x축에 리밸런싱 날짜 점선 추가 (검은색 점선)
for r_date in rebalanced_date:
    ax1.axvline(x=r_date, color='black', linestyle=':', linewidth=0.8)

# total_value 플롯 (빨간색 선)
ax2 = ax1.twinx()
ax2.plot(date, backtest, color='red', label='Total Value (Portfolio)')
ax2.set_ylabel('Value', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# x축 포맷팅: 대표값만 몇 개 출력
ax1.xaxis.set_major_locator(mdates.YearLocator())  # 매년 대표값 출력
ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))  # 1월과 7월 표시
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 주요 틱 라벨 포맷
ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%Y-%m'))  # 부 틱 라벨 포맷

# 비교지수 플롯 
ax2.plot(date, long_only, color='green', alpha=0.5, label='Long Only')


# x축 레이블 회전
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 레전드 수집 및 설정
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 그래프 타이틀 및 레전드
fig.suptitle('Back Test Result')
fig.tight_layout()
plt.show()


# 무위험 수익률 (연간 4%)
risk_free_rate_annual = 0.04
risk_free_rate_daily = (1 + risk_free_rate_annual) ** (1 / 252) - 1  # 252 거래일 기준

# 일별 수익률 계산
backtest_returns = backtest.pct_change().dropna()  # 백테스트 수익률
long_only_returns = long_only.pct_change().dropna()  # Long Only 수익률

# 초과 수익률 계산
backtest_excess_returns = backtest_returns - risk_free_rate_daily
long_only_excess_returns = long_only_returns - risk_free_rate_daily

# 샤프지수 계산
backtest_sharpe = backtest_excess_returns.mean() / backtest_excess_returns.std() * np.sqrt(252)
long_only_sharpe = long_only_excess_returns.mean() / long_only_excess_returns.std() * np.sqrt(252)

print(f"Backtest Sharpe Ratio: {backtest_sharpe:.2f}")
print(f"Long Only Sharpe Ratio: {long_only_sharpe:.2f}")