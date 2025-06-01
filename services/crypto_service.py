from pycoingecko import CoinGeckoAPI
import pandas as pd
from typing import Dict, List, Tuple
from models.crypto_model import CryptoModel
from utils.data_processor import DataProcessor
import os
import numpy as np

class CryptoService:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.model = CryptoModel()
        self.data_processor = DataProcessor()
        self.basic_features = ['price', 'volume', 'market_cap']

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

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 데이터 누락 랜덤으로 발생
        df = self.data_processor.introduce_missing_data(df, self.basic_features)
        
        # 누락된 데이터 채우기 -> 전/후 20일 평균치 계산
        for feature in self.basic_features:
            df = self.data_processor.fill_missing_data(df, feature)
        
        # 추가 피처 계산
        df = self.data_processor.calculate_features(df)
        
        return df

    def get_chart_data(self) -> Dict:
        df = self.fetch_data()
        df = self.process_data(df)

        # 예측값 인코딩 (1일 후 기준)
        model = CryptoModel()
        X_scaled, targets = model.prepare_data(df.copy())
        model.train(X_scaled, targets)
        predicted = model.model.predict(X_scaled)
        df = df.iloc[-len(predicted):]  # dropna로 줄어든 행에 맞춤
        df['predicted_trend'] = predicted  # 0: 하락/유지, 1: 상승

        # CSV 파일로 저장
        self.save_to_csv(df)

        # 1일 단위로 데이터 리샘플링
        daily_df = df.resample('D').agg({
            'price': 'mean',  
            'volume': 'sum',  
            'market_cap': 'last',  
            'fg_index': 'mean',
            'predicted_trend': 'last'  # 일별 마지막 예측값
        })

        return {
            'timestamps': daily_df.index.tolist(),
            'prices': daily_df['price'].round(2).tolist(),
            'volumes': daily_df['volume'].round(2).tolist(),
            'market_caps': daily_df['market_cap'].round(2).tolist(),
            'fg_index': daily_df['fg_index'].round(2).tolist(),
            'predicted_trend': daily_df['predicted_trend'].astype(int).tolist()
        }

    def get_fg_index(self) -> Dict:
        df = self.fetch_data()
        df = self.process_data(df)
        
        return {
            'current_fg_index': float(df['fg_index'].iloc[-1]),
            'timestamp': df.index[-1].isoformat()
        }

    def predict_price(self, period: str) -> Dict:
        df = self.fetch_data()
        df = self.process_data(df)
        
        X_scaled, targets = self.model.prepare_data(df)
        self.model.train(X_scaled, targets)
        
        latest_data = X_scaled[-1].reshape(1, -1)
        prediction, confidence = self.model.predict(latest_data, period)
        
        return {
            'prediction': bool(prediction),
            'confidence': confidence,
            'timestamp': df.index[-1].isoformat()
        } 

    def save_to_csv(self, df: pd.DataFrame, filename: str = 'crypto_data.csv'):
        filepath = os.path.join(os.getcwd(), filename)
        df.to_csv(filepath)

        