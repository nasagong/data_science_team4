## ⚡️ Quick Colab Demo

<details>

<summary> 펼치기 </summary>

```py
!pip install pycoingecko pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pycoingecko import CoinGeckoAPI
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix


class CryptoAnalyzer:
    # 분석 도구 초기화
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.scaler = StandardScaler()
        self.model = LogisticRegression()
        self.features = ['price', 'volume', 'market_cap', 'ema_14', 'rsi_14',
                         'pct_price', 'pct_volume', 'volatility', 'fg_index']
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 비트코인 가격 + 거래량 + 시가총액 데이터를 CoinGecko에서 가져온 후 Pandas로 가공 및 반환
    def fetch_data(self, days: int = 60) -> pd.DataFrame:
        data = self.cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=days)

        df_price = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df_volume = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        df_marketcap = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])

        for df in [df_price, df_volume, df_marketcap]:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        df = df_price.merge(df_volume, on='timestamp').merge(df_marketcap, on='timestamp')
        df.set_index('timestamp', inplace=True)

        return df

    # 지표 계산 및 결측값 처리 등을 통해 데이터 전처리 수행
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.introduce_missing_data(df, ['price', 'volume', 'market_cap'])
        for feature in ['price', 'volume', 'market_cap']:
            df = self.fill_missing_data(df, feature)

        df['ema_14'] = df['price'].ewm(span=14, adjust=False).mean()
        df['rsi_14'] = self.compute_rsi(df['price'])
        df['pct_price'] = df['price'].pct_change() * 100
        df['pct_volume'] = df['volume'].pct_change() * 100
        df['volatility'] = df['price'].rolling(window=24).std()
        df['volatility_pct'] = (df['volatility'] / df['price']) * 100

        a, b, c = 0.4, 0.3, 0.1
        df['fg_index'] = 50 + a * df['pct_price'] + b * df['pct_volume'] - c * df['volatility_pct']
        df['fg_index'] = df['fg_index'].clip(lower=0, upper=100)

        return df

    # 예측 타겟 생성 + 인코딩 및 특성 스케일링 수행
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        df['price_1h'] = df['price'].shift(-1)
        df['price_1d'] = df['price'].shift(-24)
        df['price_7d'] = df['price'].shift(-168)

        df['target_1h'] = (df['price_1h'] > df['price']).astype(int)
        df['target_1d'] = (df['price_1d'] > df['price']).astype(int)
        df['target_7d'] = (df['price_7d'] > df['price']).astype(int)

        df.dropna(inplace=True)

        X = df[self.features]
        X_scaled = self.scaler.fit_transform(X)

        targets = {
            '1h': df['target_1h'].values,
            '1d': df['target_1d'].values,
            '7d': df['target_7d'].values
        }

        return X_scaled, targets, df

    # 로지스틱 회귀로 각 예측 타겟에 대해 교차 검증 및 혼동 행렬 시각화
    def train_and_evaluate(self, X: np.ndarray, targets: dict, df: pd.DataFrame):
        results = {}
        for period, y in targets.items():
            print(f"\n{period} 예측 모델 평가:")

            scores = cross_val_score(self.model, X, y, cv=self.kf)
            print(f"교차 검증 점수: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

            self.model.fit(X, y)
            y_pred = self.model.predict(X)

            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {period} Prediction')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()

            results[period] = {
                'cv_score': scores.mean(),
                'cv_std': scores.std(),
                'predictions': y_pred
            }

        return results

    def visualize_results(self, df: pd.DataFrame, results: dict):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['price'], label='Actual Price')
        plt.title('Bitcoin Price Trend')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['fg_index'], label='Fear & Greed Index', color='orange')
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
        plt.title('Fear & Greed Index Trend')
        plt.xlabel('Date')
        plt.ylabel('Index')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['rsi_14'], label='RSI(14)', color='purple')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        plt.title('RSI(14) Technical Indicator')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['volatility_pct'], label='Volatility(%)', color='brown')
        plt.title('Price Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 시각화 
    def visualize_accuracy_timeseries(self, df: pd.DataFrame, results: dict, window: int = 24):
      for period, data in results.items():
        y_true = df[f'target_{period}'].values
        y_pred = data['predictions']
        accuracy_series = (y_true == y_pred).astype(int)

        rolling_acc = pd.Series(accuracy_series, index=df.index).rolling(window=window).mean()

        plt.figure(figsize=(15, 4))
        plt.plot(rolling_acc.index, rolling_acc.values, label='Rolling Accuracy', color='blue')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='50% baseline')
        plt.title(f'{period} Rolling Accuracy (Window = {window})')
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    # 과매도/과매수 지수 계산
    @staticmethod
    def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # 금융 정보 API 특성상 유의미한 결측치가 발생하지 않기에 임의로 누락값 생성
    @staticmethod
    def introduce_missing_data(df: pd.DataFrame, features: list, missing_prob: float = 0.005) -> pd.DataFrame:
        df_copy = df.copy()
        for feature in features:
            mask = np.random.random(len(df)) < missing_prob
            df_copy.loc[mask, feature] = np.nan
        return df_copy

    # 누락값을 주변 20시간 평균으로 대체
    @staticmethod
    def fill_missing_data(df: pd.DataFrame, feature: str) -> pd.DataFrame:
        df_copy = df.copy()
        missing_indices = df_copy[df_copy[feature].isna()].index

        for idx in missing_indices:
            current_pos = df_copy.index.get_loc(idx)
            start_pos = max(0, current_pos - 10)
            end_pos = min(len(df_copy), current_pos + 10)

            if end_pos - start_pos < 20:
                if start_pos == 0:
                    end_pos = min(len(df_copy), 20)
                elif end_pos == len(df_copy):
                    start_pos = max(0, len(df_copy) - 20)

            surrounding_data = df_copy.iloc[start_pos:end_pos][feature].dropna()
            if not surrounding_data.empty:
                df_copy.loc[idx, feature] = surrounding_data.mean()

        return df_copy


def run_analysis(days: int = 60):
    analyzer = CryptoAnalyzer()
    df = analyzer.fetch_data(days)
    df = analyzer.process_data(df)
    X_scaled, targets, df = analyzer.prepare_data(df)
    results = analyzer.train_and_evaluate(X_scaled, targets, df)
    analyzer.visualize_results(df, results)
    analyzer.visualize_accuracy_timeseries(df, results)
    return df, results


if __name__ == "__main__":
    df, results = run_analysis()

```
</details>

## 주요 기능

### 0. 데이터 수집 및 전처리
- [CoinGecko API](https://www.coingecko.com/en/api)를 통한 실시간 비트코인 데이터 수집
- 가격, 거래량, 시가총액 데이터 통합
- 누락값 처리 및 기술적 지표 계산

```python
def fetch_data(self, days: int = 60) -> pd.DataFrame:
    data = self.cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=days)
    df_price = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df_volume = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    df_marketcap = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
```

### 2. 기술적 지표 분석
- RSI(14): 과매수/과매도 판단
- EMA(14): 지수이동평균
- 변동성 지표
- 공포/탐욕 지수 계산

```python
def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
    df['ema_14'] = df['price'].ewm(span=14, adjust=False).mean()
    df['rsi_14'] = self.compute_rsi(df['price'])
    df['volatility'] = df['price'].rolling(window=24).std()
    df['fg_index'] = 50 + a * df['pct_price'] + b * df['pct_volume'] - c * df['volatility_pct']
```

### 3. 가격 예측 모델
- 1시간, 1일, 7일 가격 상승/하락 예측
- 로지스틱 회귀 모델 사용
- 5-fold 교차 검증을 통한 모델 평가

```python
def train_and_evaluate(self, X: np.ndarray, targets: dict, df: pd.DataFrame):
    for period, y in targets.items():
        scores = cross_val_score(self.model, X, y, cv=self.kf)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
```

### 4. 시각화
- 가격 추이 차트
- 공포/탐욕 지수 추이
- RSI 및 변동성 지표
- 예측 모델의 혼동 행렬

```python
def visualize_results(self, df: pd.DataFrame, results: dict):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['price'], label='Actual Price')
    plt.title('Bitcoin Price Trend')
```

## 사용 방법

1. 필요한 패키지 설치:
```bash
pip install pycoingecko pandas numpy matplotlib seaborn scikit-learn
```

2. 분석 실행:
```python
from crypto_analysis import run_analysis

# 60일 데이터로 분석 실행
df, results = run_analysis()

# 또는 다른 기간으로 분석 실행
df, results = run_analysis(days=30)
```

## 주요 특징
- 실시간 데이터 수집 및 분석
- 다양한 기술적 지표 활용
- 다중 기간 예측 (1시간, 1일, 7일)
- 직관적인 시각화 제공
- 모델 성능 평가 및 검증 

## 사용 방법 - Colab Demo

리드미 최상단의 Colab Quick Start를 복사해
[Colab](https://colab.google/) 환경에서 바로 실행

## 사용 방법 - API 서버

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 서버 실행
```bash
python main.py
```
서버가 `http://localhost:8000`에서 실행됩니다.

### 3. API 엔드포인트 테스트

#### 차트 데이터 조회
```bash
curl http://localhost:8000/api/chart
```

#### 가격 예측
```bash
# 1시간 예측
curl http://localhost:8000/api/predict/1h

# 1일 예측
curl http://localhost:8000/api/predict/1d

# 7일 예측
curl http://localhost:8000/api/predict/7d
```

#### 공포/탐욕 지수 조회
```bash
curl http://localhost:8000/api/fgindex
```
