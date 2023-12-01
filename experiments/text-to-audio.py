from audio_datasets.audio_dataset import ESC50Dataset, GTZANDataset, UrbanSound8kDataset
from transformers import ClapModel, ClapProcessor
import torch
from torch.nn.functional import cosine_similarity
from tqdm.auto import tqdm

# if you have CUDA or MPS, set it to the active device like this
# device = "cuda" if torch.cuda.is_available() else \
#          ("mps" if torch.backends.mps.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the ESC50Dataset
# metadata_path = 'audios/ESC-50/ESC-50-master/meta/esc50.csv'
# audio_dir = 'audios/ESC-50/ESC-50-master/audio/'
# dataset = ESC50Dataset(metadata_path, audio_dir)

# Initialize the GTZANDataset
metadata_path = 'audios/GTZAN/features_30_sec.csv'
audio_dir = 'audios/GTZAN/genres_original/'
dataset = GTZANDataset(metadata_path, audio_dir)


# model_id = "laion/clap-htsat-fused"
model_id = "laion/larger_clap_music"

# we initialize model and audio processor
processor = ClapProcessor.from_pretrained(model_id)
model = ClapModel.from_pretrained(model_id).to(device)

# Process a text query
texts = ['A chill jazz song']
inputs_text = processor(text=texts, return_tensors="pt", padding=True)
inputs_text = {key: value.to(device) for key, value in inputs_text.items()}

# Get the text embeddings
with torch.inference_mode():
    outputs_text = model.get_text_features(**inputs_text)


# Get a list of all items in the dataset
all_items = list(range(len(dataset)+1))
#all_items = list(range(16))
# sample_idx = np.random.randint(0, len(dataset)+1, 100).tolist()
# sample_idx = list(range(16))
# Randomly select 100 items
# selected_items = random.sample(all_items, 100)

# Get the audio embeddings
audio_embeddings = []
for idx in tqdm(all_items, "processing audio files"):
    waveform, _ = dataset[idx]
    inputs_audio = processor(audios=waveform, sampling_rate=48000, return_tensors="pt", padding=True)
    inputs_audio = {key: value.to(device) for key, value in inputs_audio.items()}
    with torch.inference_mode():
        outputs_audio = model.get_audio_features(**inputs_audio)
    audio_embeddings.append(outputs_audio)


# Assume `outputs_text` is your text embedding and `audio_embeddings` is a list of audio embeddings
#text_embedding = outputs_text.squeeze(0)  # Remove the batch dimension
text_embedding = outputs_text
# Calculate cosine similarity
# similarities = [cosine_similarity(text_embedding, audio_embedding.squeeze(0)) for audio_embedding in audio_embeddings]
similarities = [cosine_similarity(text_embedding, audio_embedding) for audio_embedding in audio_embeddings]
# Get the indices of the audio files with the highest similarity
top_indices = torch.topk(torch.tensor(similarities), k=5).indices  # Get top 5 indices
# print(top_indices)

# Print the top 5 most similar audio files and their similarity scores
for idx in top_indices:
    print(f"Cosine Similarity: {similarities[idx]}")
    print(f"Audio file: {dataset.filenames[idx]}")
    print(f"Label: {dataset.labels[idx]}")
    print(f"Index: {idx}")
    print("")

