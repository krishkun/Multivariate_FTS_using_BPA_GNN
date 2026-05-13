"""
Data Provider Module for Fuzzy BPA EGNN

This module provides data loading and preprocessing for:
- Electricity dataset
- Traffic dataset
- Weather dataset
- ETT (Electricity Transformer Temperature) dataset
- Exchange dataset
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        stride: int = 1,
    ):
        """
        Args:
            data: Time series data of shape (num_samples, num_features)
            seq_len: Input sequence length
            pred_len: Prediction horizon
            stride: Stride for sliding window
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        
        # Compute number of samples
        self.num_samples = (len(data) - seq_len - pred_len) // stride + 1
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        
        x = self.data[start_idx:end_idx]
        y = self.data[end_idx:end_idx + self.pred_len]
        
        return x, y


class DataProvider:
    """
    Data provider for time series datasets.
    
    Supports:
    - Electricity: 321 variables, hourly data
    - Traffic: 862 variables, hourly data
    - Weather: 21 variables, 10-minute data
    - ETTh1/ETTh2: 7 variables, hourly data
    - ETTm1/ETTm2: 7 variables, 15-minute data
    - Exchange: 8 variables, daily data
    """
    
    DATASET_INFO = {
        'Electricity': {'num_features': 321, 'freq': 'h'},
        'Traffic': {'num_features': 862, 'freq': 'h'},
        'Weather': {'num_features': 21, 'freq': '10min'},
        'ETTh1': {'num_features': 7, 'freq': 'h'},
        'ETTh2': {'num_features': 7, 'freq': 'h'},
        'ETTm1': {'num_features': 7, 'freq': '15min'},
        'ETTm2': {'num_features': 7, 'freq': '15min'},
        'Exchange': {'num_features': 8, 'freq': 'd'},
    }
    
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        seq_len: int = 96,
        pred_len: int = 96,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        scaler_type: str = 'standard',
    ):
        """
        Args:
            data_path: Path to data directory
            dataset_name: Name of dataset
            seq_len: Input sequence length
            pred_len: Prediction horizon
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Get dataset info
        if dataset_name not in self.DATASET_INFO:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        self.num_features = self.DATASET_INFO[dataset_name]['num_features']
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self) -> np.ndarray:
        """Load dataset from file."""
        file_path = os.path.join(self.data_path, f"{self.dataset_name}.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            # Try alternative file extensions
            for ext in ['.csv', '.txt', '.dat']:
                alt_path = os.path.join(self.data_path, f"{self.dataset_name}{ext}")
                if os.path.exists(alt_path):
                    df = pd.read_csv(alt_path)
                    break
            else:
                raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        # Remove date column if present
        if 'date' in df.columns:
            df = df.drop('date', axis=1)
        
        return df.values
    
    def _split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets."""
        n = len(self.data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_data = self.data[:train_end]
        val_data = self.data[train_end:val_end]
        test_data = self.data[val_end:]
        
        return train_data, val_data, test_data
    
    def get_data(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for train, validation, and test.
        
        Args:
            batch_size: Batch size
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_data, val_data, test_data = self._split_data()
        
        # Fit scaler on training data
        self.scaler.fit(train_data)
        
        # Transform data
        train_data = self.scaler.transform(train_data)
        val_data = self.scaler.transform(val_data)
        test_data = self.scaler.transform(test_data)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(train_data, self.seq_len, self.pred_len)
        val_dataset = TimeSeriesDataset(val_data, self.seq_len, self.pred_len)
        test_dataset = TimeSeriesDataset(test_data, self.seq_len, self.pred_len)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        return self.scaler.inverse_transform(data)


class DatasetDownloader:
    """Download datasets from various sources."""
    
    DATASET_URLS = {
        'Electricity': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2011.txt.zip',
        'Traffic': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00481/PEMS-SF.zip',
        # Add more URLs as needed
    }
    
    @staticmethod
    def download(dataset_name: str, save_path: str):
        """Download a dataset."""
        import urllib.request
        import zipfile
        
        if dataset_name not in DatasetDownloader.DATASET_URLS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        url = DatasetDownloader.DATASET_URLS[dataset_name]
        os.makedirs(save_path, exist_ok=True)
        
        zip_path = os.path.join(save_path, f"{dataset_name}.zip")
        
        print(f"Downloading {dataset_name}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print(f"Extracting {dataset_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_path)
        
        os.remove(zip_path)
        print(f"Downloaded {dataset_name} to {save_path}")


def create_data_provider(
    data_path: str,
    dataset_name: str,
    seq_len: int = 96,
    pred_len: int = 96,
    **kwargs,
) -> DataProvider:
    """Factory function to create data provider."""
    return DataProvider(
        data_path=data_path,
        dataset_name=dataset_name,
        seq_len=seq_len,
        pred_len=pred_len,
        **kwargs,
    )


if __name__ == '__main__':
    # Example usage
    data_path = './dataset'
    dataset_name = 'ETTh1'
    
    provider = DataProvider(
        data_path=data_path,
        dataset_name=dataset_name,
        seq_len=96,
        pred_len=96,
    )
    
    train_loader, val_loader, test_loader = provider.get_data(batch_size=32)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check a batch
    for x, y in train_loader:
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")
        break
