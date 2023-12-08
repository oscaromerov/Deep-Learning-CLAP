import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa

# Defining which device to use (mps only available on silicon chips)
# device = "cuda" if torch.cuda.is_available() else \
#          ("mps" if torch.backends.mps.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

class ESC50Dataset(Dataset):
    def __init__(self, metadata_path, audio_dir):
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.df = pd.read_csv(metadata_path)
        self.labels = self.df['category'].values
        self.filenames = self.df['filename'].values
        self.classes = self.df['category'].unique()

    def get_waveform(self, idx):
        audio_path = os.path.join(self.audio_dir, self.filenames[idx])
        # Check if the file exists
        if not os.path.exists(audio_path):
            raise RuntimeError(f"File not found: {audio_path}")

        waveform, _ = librosa.load(audio_path, sr=None)
        return waveform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform = self.get_waveform(idx)
        label = self.labels[idx]
        return waveform, label


class GTZANDataset(Dataset):
    def __init__(self, metadata_path, audio_dir):
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.df = pd.read_csv(metadata_path)
        self.labels = self.df['label'].values
        self.filenames = self.df['filename'].values
        self.classes = self.df['label'].unique()
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = os.path.join(self.audio_dir, self.labels[idx],self.filenames[idx])
        waveform, _ = librosa.load(filepath, sr=None)
        label = self.labels[idx]
        return waveform, label

class UrbanSound8kDataset(Dataset):
    def __init__(self, metadata_path, audio_dir):
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.df = pd.read_csv(metadata_path)
        self.labels = self.df['class'].values
        self.filenames = self.df['slice_file_name'].values
        self.classes = self.df['class'].unique()
        self.folds = self.df['fold'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = os.path.join(self.audio_dir, f"fold{self.folds[idx]}", self.filenames[idx])
        waveform, _ = librosa.load(filepath, sr=None)
        label = self.labels[idx]
        return waveform, label


class CustomAudioDataset(Dataset):
    def __init__(self, audio_dir, metadata_path):
        self.audio_dir = audio_dir
        self.metadata_path = metadata_path
        self.df = pd.read_csv(metadata_path)
        self.labels = self.df['Sentiment'].values  # Adjust the column name based on your metadata
        self.filenames = self.df['Audio_File'].values  # Adjust the column name based on your metadata
        self.classes = self.df['Audio_File'].unique()

    def get_waveform(self, idx):
        audio_path = os.path.join(self.audio_dir, self.filenames[idx])
        if not os.path.exists(audio_path):
            raise RuntimeError(f"File not found: {audio_path}")

        waveform, _ = librosa.load(audio_path, sr=None)
        return waveform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform = self.get_waveform(idx)
        label = self.labels[idx]
        return waveform, label

# Usage example:
audio_dir = '/Users/fidenciofernandez/OneDrive/IMT/M1S2/DeepLearning/Proj/Epic_folder'
metadata_path = '/Users/fidenciofernandez/OneDrive/IMT/M1S2/DeepLearning/Proj/Epic_folder/epic_music_labels.csv'  # Make sure your metadata has 'label' and 'filename' columns

custom_dataset = CustomAudioDataset(audio_dir, metadata_path)

# Get the first sample and unpack
first_data = custom_dataset[0]
features, label = first_data
print(features, label)

#
#
#
# # create dataset
# esc50_csv = 'audios/ESC-50/ESC-50-master/meta/esc50.csv'
# esc50_audio_dir = 'audios/ESC-50/ESC-50-master/audio/'
# dataset = ESC50Dataset(esc50_csv, esc50_audio_dir)
# # print(len(dataset))
# # # #dataset = ESC_50(esc50_audio_dir, esc50_csv)
# # #
# # # get first sample and unpack
# first_data = dataset[65]
# features, labels = first_data
# print(features, labels)
# print(features.shape)
# print(torch.dtype(features))
# print(labels.shape)
#
# # create dataset GTZAN
# gtzan_csv = 'audios/GTZAN/features_30_sec.csv'
# gtzan_audio_dir = 'audios/GTZAN/genres_original'
# dataset = GTZANDataset(gtzan_csv, gtzan_audio_dir)
# #dataset = ESC_50(esc50_audio_dir, esc50_csv)
#
# # get first sample and unpack
# first_data = dataset[679]
# features, labels = first_data
# print(features, labels)

# # create dataset URBANSOUND8K
# urban_csv = 'audios/UrbanSound8K/metadata/UrbanSound8K.csv'
# urban_audio_dir = 'audios/UrbanSound8K/audio/'
# dataset = UrbanSound8kDataset(urban_csv, urban_audio_dir)
# #dataset = ESC_50(esc50_audio_dir, esc50_csv)

# # get first sample and unpack
# first_data = dataset[10]
# features, labels = first_data
# print(features, labels)
