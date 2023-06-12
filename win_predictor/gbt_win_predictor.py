import numpy as np
import pandas as pd
import xgboost as xgb

from win_predictor.win_feature_extractor import WinFeatureExtractor
from win_predictor.win_predictor import WinPredictor


class GBTWinPredictor(WinPredictor):
    model: xgb.XGBClassifier
    feature_extractor: WinFeatureExtractor

    def __init__(self, feature_extractor: WinFeatureExtractor):
        self.model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.3)
        self.feature_extractor = feature_extractor

    def train(self, x: pd.DataFrame, y: np.ndarray) -> None:
        features = self.feature_extractor.get_features(x)
        self.model.fit(features, y)

    def predict(self, x: pd.DataFrame) -> [int]:
        features = self.feature_extractor.get_features(x)
        return self.model.predict(features)
