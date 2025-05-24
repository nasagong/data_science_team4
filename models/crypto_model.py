from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

class CryptoModel:
    def __init__(self):
        self.scaler = StandardScaler()

        # 상승/하락 여부 분류 위헤 로지스틱 회귀 채택
        self.model = LogisticRegression()
        self.features = ['price', 'volume', 'market_cap', 'ema_14', 'rsi_14',
                        'pct_price', 'pct_volume', 'volatility', 'fg_index']
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.model_scores = {}

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        # 타겟값 생성
        df['price_1h'] = df['price'].shift(-1)
        df['price_1d'] = df['price'].shift(-24)
        df['price_7d'] = df['price'].shift(-168)

        df['target_1h'] = (df['price_1h'] > df['price']).astype(int)
        df['target_1d'] = (df['price_1d'] > df['price']).astype(int)
        df['target_7d'] = (df['price_7d'] > df['price']).astype(int)

        df.dropna(inplace=True)

        # 스케일링
        # 모든 피처를 평균 0, 표준편차 1로 정규화
        X = df[self.features]
        X_scaled = self.scaler.fit_transform(X)

        # 타겟값 준비 (임시값이라 최종 데이터에는 포함되지 않습니다 루트 경로 최종 csv 확인해주세요)
        targets = {
            '1h': df['target_1h'].values,
            '1d': df['target_1d'].values,
            '7d': df['target_7d'].values
        }

        return X_scaled, targets

    # 학습 메서드
    def train(self, X: np.ndarray, targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        for period, y in targets.items():
            # 교차 검증 먼저
            scores = cross_val_score(self.model, X, y, cv=self.kf)
            self.model_scores[period] = round(scores.mean(), 4)
            
            # 최종 모델 학습
            self.model.fit(X, y)
        return self.model_scores

    # 예측 메서드
    def predict(self, X: np.ndarray, period: str) -> Tuple[int, float]:
        if period not in self.model_scores:
            raise ValueError(f"Invalid period: {period}. Must be one of ['1h', '1d', '7d']")
        
        prediction = self.model.predict(X)
        confidence = self.model_scores[period]
        
        return int(prediction[0]), confidence 