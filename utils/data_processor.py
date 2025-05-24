import pandas as pd
import numpy as np
from typing import List, Tuple
import random

class DataProcessor:
    # 과매수 / 과매도 여부 계산
    @staticmethod
    def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # 기본 피처 누락값 0.5% 확률로 생성
    @staticmethod
    def introduce_missing_data(df: pd.DataFrame, features: List[str], missing_prob: float = 0.005) -> pd.DataFrame:
        df_copy = df.copy()
        for feature in features:
            mask = np.random.random(len(df)) < missing_prob
            df_copy.loc[mask, feature] = np.nan
        return df_copy

    # 20시간 평균 치 사용하여 누락값 복원원
    @staticmethod
    def fill_missing_data(df: pd.DataFrame, feature: str) -> pd.DataFrame:
        df_copy = df.copy()
        missing_indices = df_copy[df_copy[feature].isna()].index
        
        for idx in missing_indices:
            # 현재 idx 위치 탐색
            current_pos = df_copy.index.get_loc(idx)
            
            # 전/후 20시간 평균치 계산하여 누락값 복원
            # => 인덱스 위치 상 균등하게 전/후 10시간씩 수집할 수 없는 상황이라면 유연하게 조정하도록 설정
            start_pos = max(0, current_pos - 10)
            end_pos = min(len(df_copy), current_pos + 10)
            
            if end_pos - start_pos < 20:
                if start_pos == 0:
                    end_pos = min(len(df_copy), 20)
                elif end_pos == len(df_copy):
                    start_pos = max(0, len(df_copy) - 20)
            
            # 평균 계산
            surrounding_data = df_copy.iloc[start_pos:end_pos][feature].dropna()
            if not surrounding_data.empty:
                df_copy.loc[idx, feature] = surrounding_data.mean()
        
        return df_copy

    # 파생 피처 계산산
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        df['ema_14'] = df['price'].ewm(span=14, adjust=False).mean()
        df['rsi_14'] = DataProcessor.compute_rsi(df['price'])
        df['pct_price'] = df['price'].pct_change() * 100
        df['pct_volume'] = df['volume'].pct_change() * 100
        df['volatility'] = df['price'].rolling(window=24).std()
        df['volatility_pct'] = (df['volatility'] / df['price']) * 100

        # gfindex 계산입니다
        a, b, c = 0.4, 0.3, 0.1
        df['fg_index'] = 50 + a * df['pct_price'] + b * df['pct_volume'] - c * df['volatility_pct']
        df['fg_index'] = df['fg_index'].clip(lower=0, upper=100)

        return df 