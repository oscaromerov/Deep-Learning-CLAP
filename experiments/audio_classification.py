from audio_datasets.audio_dataset import ESC50Dataset, GTZANDataset, UrbanSound8kDataset
from transformers import ClapModel, ClapProcessor
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.functional import cosine_similarity
from tqdm.auto import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the ESC50Dataset
metadata_path = 'audios/ESC-50/ESC-50-master/meta/esc50.csv'
audio_dir = 'audios/ESC-50/ESC-50-master/audio/'
dataset = ESC50Dataset(metadata_path, audio_dir)

classes = dataset.classes # Get the classes
# print(classes)

classes_sentences = [f"This is a sound of {c}" for c in classes]  # Create a sentence for each class
# print(classes_sentences)

model_id = "laion/clap-htsat-fused"

processor = ClapProcessor.from_pretrained(model_id)
model = ClapModel.from_pretrained(model_id).to(device)

# Get text embeddings
inputs_text = processor(text=classes_sentences, return_tensors="pt", padding=True)
inputs_text = {key: value.to(device) for key, value in inputs_text.items()}
with torch.inference_mode():
    outputs_text = model.get_text_features(**inputs_text)

# Get text embeddings
text_emb = outputs_text.detach().cpu().numpy()

predictions = []
batch_size: int = 2

# len_dataset = len(dataset)
len_dataset = 16

for i in tqdm(range(0, len_dataset, batch_size)):
    i_end = min(i + batch_size, len_dataset)
    audios = [dataset[j][0] for j in range(i, i_end)]
    inputs_audio = processor(audios=audios, sampling_rate=48000, return_tensors="pt", padding=True)
    inputs_audio = {key: value.to(device) for key, value in inputs_audio.items()}
    with torch.inference_mode():
        outputs_audio = model.get_audio_features(**inputs_audio)
    audio_emb = outputs_audio.detach().cpu().numpy()
    assert audio_emb.shape[1] == text_emb.shape[1], "Audio and text embeddings must have the same dimension"
    # Calculate scores
    scores = np.dot(audio_emb, text_emb.T)
    # scores = np.dot(audio_emb, text_emb.T.cpu().numpy())
    predictions.extend(np.argmax(scores, axis=1))

true_predictions = []
for i in range(len_dataset):
    waveform, label = dataset[i]
    print(f"True label: {label}, Pred label: {classes[predictions[i]]}, filename: {dataset.filenames[i]}")
    if classes[predictions[i]] == label:
        true_predictions.append(1)
    else:
        true_predictions.append(0)

accuracy = sum(true_predictions) / len(true_predictions)
print(f"Accuracy: {accuracy}")

# results = zip(true_predictions, predictions)
# for i, (true, pred) in enumerate(results):
#     print(f"True: {classes[true]}, Pred: {classes[pred]}, Index: {i}")
# Create a subset of the first 16 samples of your dataset
# subset = Subset(dataset, range(16))
#
# batch_size = 2
# dataloader = DataLoader(subset, batch_size=batch_size)
#
# # Initialize your variables
# preds = []
# true_preds = []
#
# # Iterate over the DataLoader
# for i, (waveforms, labels) in enumerate(dataloader):
#     # Process the batch of waveforms
#     inputs_audio = processor(audios=waveforms, sampling_rate=48000, return_tensors="pt", padding=True)
#     inputs_audio = {key: value.to(device) for key, value in inputs_audio.items()}
#     with torch.inference_mode():
#         outputs_audio = model.get_audio_features(**inputs_audio)
#     audio_emb = outputs_audio.detach().cpu().numpy()
#
#     # Calculate scores and predictions
#     scores = np.dot(audio_emb, text_emb.T)
#     batch_preds = np.argmax(scores, axis=1)
#     preds.extend(batch_preds)
#
#     # Compare predictions to true labels
#     for j in range(len(labels)):
#         print(f"True label: {labels[j]}, Pred label: {classes[batch_preds[j]]}, filename: {dataset.get_filename(j)}")
#         if classes[batch_preds[j]] == labels[j]:
#             true_preds.append(1)
#         else:
#             true_preds.append(0)
#
# # Calculate accuracy
# accuracy = sum(true_preds) / len(true_preds)
# print(f"Accuracy: {accuracy}")


