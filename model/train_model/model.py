import librosa
import librosa.display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix


class InstrumentDataset(Dataset):
    def __init__(self, dataframe, onehot, target_sr=22050, window_size=0.5, hop_size=0.25, n_mels=128):
        """
        dataframe: Pandas DataFrame containing 'filepath' and 'Class' columns.
        onehot: OneHotEncoder instance for encoding class labels.
        target_sr: Target sample rate (default 44.1kHz).
        window_size: Window size in seconds (default 2 sec).
        hop_size: Step size between windows in seconds (default 1 sec).
        n_mels: Number of Mel bands.
        """
        self.file_list = dataframe["fname"].values
        self.labels = dataframe["Class"].values
        # self.duration = dataframe["duration"].values
        
        self.onehot = onehot
        self.target_sr = target_sr
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_mels = n_mels

        self.data = self._process_audio_files()
    
    def _audio_to_melspectrogram(self, audio, sr, n_mels):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB
        return mel_spec_db

    def _process_audio_files(self):
        processed_data = []
        for file_path, label in zip(self.file_list, self.labels):
            y, sr = librosa.load(file_path, sr=self.target_sr)  # Load & resample
            y, _ = librosa.effects.trim(y, top_db=10) # Removes initial and end silence
            y = librosa.util.normalize(y)  # Normalize waveform
            num_samples = int(self.window_size * sr)  # Convert window size to samples
            hop_samples = int(self.hop_size * sr)  # Convert hop size to samples
            
            encoded_label = self.onehot.transform([[label]])[0]  # One-hot encode

            # Apply sliding window
            for start in range(0, len(y) - num_samples, hop_samples):
                window = y[start:start + num_samples]
                if len(window) < num_samples: # Skip incomplete window
                    break
                mel_spec_db = self._audio_to_melspectrogram(window, sr, self.n_mels)
                processed_data.append((mel_spec_db, encoded_label))
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_spec_db, label = self.data[idx]
        mel_spec_db = torch.tensor(mel_spec_db).unsqueeze(0)  # Add channel dimension
        return mel_spec_db.float(), torch.tensor(label).float()  # Ensure label is float for loss calculation
    

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        reduced_channels = max(1, in_channels // reduction)  # Avoid division errors

        # Channel Attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        ch_att = self.channel_att(x) * x

        # Spatial Attention
        avg_pool = torch.mean(ch_att, dim=1, keepdim=True)
        max_pool, _ = torch.max(ch_att, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sp_att = self.spatial_att(spatial_input) * ch_att

        return sp_att


# Model with CBAM
class InstrumentClassifier_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(InstrumentClassifier_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.cbam1 = CBAM(8)  # CBAM after first conv layer
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.cbam2 = CBAM(16)  # CBAM after second conv layer
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.cbam3 = CBAM(32)  # CBAM after third conv layer
        
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 5 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Convolutional layers with CBAM
        x = self.pool(F.relu(self.conv1(x)))  # (11, 64, 8)
        x = self.cbam1(x)
        x = self.bn1(x)

        x = self.pool(F.relu(self.conv2(x)))  # (5, 32, 16)
        x = self.cbam2(x)
        x = self.bn2(x)

        x = F.relu(self.conv3(x))  # (5, 32, 32)
        x = self.cbam3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x



class InstrumentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(InstrumentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # Conv Layer 1
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # Conv Layer 2
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Conv Layer 3
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        self.fc1 = nn.Linear(32 * 5 * 32, 1024)  # Fully connected layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)  # Output layer

    def forward(self, x):
        # 22 x 128
        x = self.pool(F.relu(self.conv1(x))) # 11 x 64 x 8
        # x = self.cbam(x)
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x))) # 5 x 32 x 16
        x = self.bn2(x)
        x = F.relu(self.conv3(x)) # 5 x 32 x 32
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


