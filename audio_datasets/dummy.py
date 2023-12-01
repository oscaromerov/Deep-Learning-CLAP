# class ESC50Dataset(Dataset):
#     def __init__(self, metadata_path, audio_dir):
#         self.metadata_path = metadata_path
#         self.audio_dir = audio_dir
#         self.df = pd.read_csv(metadata_path)
#         self.labels = self.df['category'].values
#         self.filenames = self.df['filename'].values
#         self.classes = self.df['category'].unique()
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         audio_path = os.path.join(self.audio_dir, self.filenames[idx])
#         # Check if the file exists
#         if not os.path.exists(audio_path):
#             raise RuntimeError(f"File not found: {audio_path}")
#
#         waveform, sample_rate = librosa.load(audio_path, sr=None)
#         # waveform = torch.from_numpy(waveform).to(device)
#         label = self.labels[idx]
#         return waveform, label
#
#     def get_class(self, idx):
#         return self.classes[idx]
#
#     def get_filename(self, idx):
#         return self.filenames[idx]
#
#     def get_label(self, idx):
#         return self.labels[idx]

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