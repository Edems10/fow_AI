from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from win_predictor.win_feature_extractor import WinFeatureExtractor
from win_predictor.win_predictor import WinPredictor


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x, self.y = torch.Tensor(x), torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class MLPWinPredictor(WinPredictor):
    device: torch.device
    model: Optional[torch.nn.Module]
    feature_extractor: WinFeatureExtractor

    def __init__(self, feature_extractor: WinFeatureExtractor):
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        # self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.device = torch.device("cpu")

        self.model = None
        self.feature_extractor = feature_extractor

    def train(self, x: pd.DataFrame, y: np.ndarray) -> None:
        features = self.feature_extractor.get_features(x)

        self.model = torch.nn.Sequential(torch.nn.Linear(features.shape[1], 4),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(4, 1))
        self.model.to(self.device)
        batch_size = 64
        dataset = SimpleDataset(features.astype(np.float32),
                                y.astype(np.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        training_steps = 300
        print_interval = 10
        steps_done = 0
        running_loss = 0.0
        while steps_done < training_steps:
            for data in loader:
                inputs, labels = data
                optimizer.zero_grad()

                outputs = self.model(inputs.to(self.device))
                loss = criterion(outputs, labels.unsqueeze(1).to(self.device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                steps_done += 1
                if steps_done % print_interval == 0:  # print every 10 mini-batches
                    print(f'[{steps_done}], loss: {running_loss / print_interval:.3f}')
                    running_loss = 0.0
                if steps_done >= training_steps:
                    break
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def predict(self, x: pd.DataFrame) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_extractor.get_features(x)
            return torch.round(torch.sigmoid(self.model(torch.Tensor(features).to(self.device))))
