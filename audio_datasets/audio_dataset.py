import os
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import sounddevice as sd


class BaseAudioDataset(Dataset):
    def get_waveform_and_sample_rate(self, idx):
        audio_path = os.path.join(self.audio_dir, self.filenames[idx])
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.audio_dir, self.labels[idx], self.filenames[idx])
            if not os.path.exists(audio_path):
                raise RuntimeError(f"File not found: {audio_path}")
        waveform, sr = librosa.load(audio_path, sr=None)
        return waveform, int(sr)

    def display_waveform(self, idx):
        waveform, sr = self.get_waveform_and_sample_rate(idx)
        time = np.arange(0, len(waveform)) / sr
        plt.figure(figsize=(14, 5))
        plt.plot(time, waveform)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.show()

    def play_audio_in_jupyter(self, idx):
        waveform, sr = self.get_waveform_and_sample_rate(idx)
        return ipd.Audio(waveform, rate=int(sr))

    def play_audio(self, idx):
        waveform, sr = self.get_waveform_and_sample_rate(idx)
        sd.play(waveform, samplerate=int(sr))
        sd.wait()

    def display_spectrogram(self, idx):
        waveform, sr = self.get_waveform_and_sample_rate(idx)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.show()


class ESC50Dataset(BaseAudioDataset):
    def __init__(self, metadata_path, audio_dir):
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.df = pd.read_csv(metadata_path)
        self.labels = self.df['category'].values
        self.filenames = self.df['filename'].values
        self.classes = self.df['category'].unique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform, _ = self.get_waveform_and_sample_rate(idx)
        label = self.labels[idx]
        return waveform, label


class GTZANDataset(BaseAudioDataset):
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
        waveform, _ = self.get_waveform_and_sample_rate(idx)
        label = self.labels[idx]
        return waveform, label


class MusicSentimentDataset(BaseAudioDataset):
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
        waveform, _ = self.get_waveform_and_sample_rate(idx)
        label = self.labels[idx]
        return waveform, label
