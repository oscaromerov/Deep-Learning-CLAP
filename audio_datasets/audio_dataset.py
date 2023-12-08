import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import numpy as np
from IPython.display import Audio, display

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
        # self.labels = self.df['label'].values[:10]
        # self.filenames = self.df['filename'].values[:10]
        self.labels = self.df['label'].values
        self.filenames = self.df['filename'].values
        self.classes = self.df['label'].unique()
    def __len__(self):
        return len(self.df)
        # return len(self.filenames)

    def __getitem__(self, idx):
        filepath = os.path.join(self.audio_dir, self.labels[idx],self.filenames[idx])
        waveform, _ = librosa.load(filepath, sr=None)
        label = self.labels[idx]
        # Ensure the audio clip is of a fixed length
        fixed_length = 661794  # Replace with the desired length
        if len(waveform) < fixed_length:
            # If the audio clip is shorter than the fixed length, pad it with zeros
            waveform = np.pad(waveform, (0, fixed_length - len(waveform)))
        elif len(waveform) > fixed_length:
            # If the audio clip is longer than the fixed length, trim it
            waveform = waveform[:fixed_length]

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
# class MusicSentimentDataset(Dataset):
#     def __init__(self, metadata_path, audio_dir):
#         self.metadata_path = metadata_path
#         self.audio_dir = audio_dir
#         self.df = pd.read_csv(metadata_path)
#         self.labels = self.df['label'].values
#         self.filenames = self.df['filename'].values
#         self.classes = self.df['label'].unique()
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         filepath = os.path.join(self.audio_dir, self.labels[idx], self.filenames[idx])
#         waveform, _ = librosa.load(filepath, sr=None)
#         label = self.labels[idx]
#         return waveform, label

## ONLY PICK SOME NUMBER OF AUDIO FILES
class MusicSentimentDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, num_audios: int = 200):
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.df = pd.read_csv(metadata_path)
        self.labels = []
        self.filenames = []
        self.classes = self.df['label'].unique()

        for class_name in self.classes:
            class_files = self.df[self.df['label'] == class_name]['filename'].values[:num_audios]
            self.filenames.extend(class_files)
            self.labels.extend([class_name] * len(class_files))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath = os.path.join(self.audio_dir, self.labels[idx], self.filenames[idx])
        waveform, _ = librosa.load(filepath, sr=None)
        label = self.labels[idx]
        return waveform, label
    #
    # def play_audio(self, idx):
    #     waveform, sr, _ = self.__getitem__(idx)
    #     # return Audio(waveform, rate=sr)
    #     print(f"SR: {sr}")
    #     return display(Audio(waveform, rate=sr))


# # # create dataset MUSICSENTIMENT
# urban_csv = 'audios/emotions/emotions_music_labels.csv'
# urban_audio_dir = 'audios/emotions/'
# dataset = MusicSentimentDataset(urban_csv, urban_audio_dir)
# #dataset = ESC_50(esc50_audio_dir, esc50_csv)
# #
# # # get first sample and unpack
# # first_data = dataset[500]
# # features, labels = first_data
# # print(features, labels)
# # dataset.play_audio(500)
# print(len(dataset))
# for i in range(len(dataset)):
#     print(dataset[i][0].shape)
#     print(dataset[i][1])
#     print(dataset.filenames[i])

metadata_path = "audios/GTZAN/features_30_sec.csv"
audio_dir = "audios/GTZAN/genres_original"
dataset = GTZANDataset(metadata_path, audio_dir)
print(f"DATASET: {len(dataset)}")

