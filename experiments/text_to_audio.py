from audio_datasets.audio_dataset import ESC50Dataset, GTZANDataset, MusicSentimentDataset
from transformers import ClapModel, ClapProcessor
import torch
from torch.nn.functional import cosine_similarity
from tqdm.auto import tqdm

class Text2Audio:
    def __init__(self, metadata_path: str, audio_dir: str, dataset_class: str,
                 model_id: str, text_query: str):
        # Set Device where model is going to run
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Define Dataset to use
        if dataset_class == 'ESC50':
            self.dataset = ESC50Dataset(metadata_path, audio_dir)
        elif dataset_class == 'GTZAN':
            self.dataset = GTZANDataset(metadata_path, audio_dir)
        elif dataset_class == 'MusicSentiment':
            self.dataset = MusicSentimentDataset(metadata_path, audio_dir)
        else:
            raise ValueError("Invalid dataset choice.")

        # Load model and processor
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(self.device)
        self.text_query = text_query
        self.inputs_text = self.processor(text=self.text_query, return_tensors="pt", padding=True)
        self.inputs_text = {key: value.to(self.device) for key, value in self.inputs_text.items()}
        with torch.inference_mode():
            self.outputs_text = self.model.get_text_features(**self.inputs_text)
        self.text_embedding = self.outputs_text

    def get_audio_embeddings(self):
        all_items = list(range(len(self.dataset)))
        audio_embeddings = []
        for idx in tqdm(all_items, "processing audio files"):
            waveform, _ = self.dataset[idx]
            inputs_audio = self.processor(audios=waveform, sampling_rate=48000, return_tensors="pt", padding=True)
            inputs_audio = {key: value.to(self.device) for key, value in inputs_audio.items()}
            with torch.inference_mode():
                outputs_audio = self.model.get_audio_features(**inputs_audio)
            audio_embeddings.append(outputs_audio)
        return audio_embeddings

    def get_similarities(self, audio_embeddings):
        similarities = [cosine_similarity(self.text_embedding, audio_embedding) for audio_embedding in audio_embeddings]
        return similarities

    def get_top_indices(self, similarities, k=5):
        top_indices = torch.topk(torch.tensor(similarities), k=k).indices
        return top_indices

#
# metadata_path = 'audios/ESC-50/ESC-50-master/meta/esc50.csv'
# audio_dir = 'audios/ESC-50/ESC-50-master/audio/'
# dataset = ESC50Dataset(metadata_path, audio_dir)
# dataset_class = 'ESC50'
# text_query = 'a dog barking'
#
# # Initialize the GTZANDataset
# # metadata_path = 'audios/GTZAN/features_30_sec.csv'
# # audio_dir = 'audios/GTZAN/genres_original/'
# # dataset = GTZANDataset(metadata_path, audio_dir)
#
#
# model_id = "laion/clap-htsat-fused"
# # model_id = "laion/larger_clap_music"
#
# # Initialize the Text2Audio
# text2audio = Text2Audio(metadata_path, audio_dir, dataset_class, model_id, text_query)
#
# # Get audio embeddings
# audio_embeddings = text2audio.get_audio_embeddings()
#
# # Get similarities
# similarities = text2audio.get_similarities(audio_embeddings)
#
# # Get top indices
# top_indices = text2audio.get_top_indices(similarities, k=5)
#
# # Print the top 5 most similar audio files and their similarity scores
# for idx in top_indices:
#     print(f"Cosine Similarity: {similarities[idx]}")
#     print(f"Audio file: {dataset.filenames[idx]}")
#     print(f"Label: {dataset.labels[idx]}")
#     print(f"Index: {idx}")
#     print("")
#
# # Display waveform, filename and play audio for the top match
# idx = top_indices[0]
# # text2audio.display_waveform(idx)
# text2audio.dataset.display_waveform(idx)
# text2audio.dataset.display_spectrogram(idx)
# text2audio.dataset.play_audio(idx)

