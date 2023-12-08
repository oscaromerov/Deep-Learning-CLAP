from audio_datasets.audio_dataset import GTZANDataset
from transformers import ClapModel, ClapProcessor
import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from text_label_augmentation import augmentations_esc
# import utils
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

# Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device available:{device}")

# Initialize GTZANDataset
metadata_path = "audios/GTZAN/features_30_sec.csv"
audio_dir = "audios/GTZAN/genres_original"
dataset_class = 'GTZAN'
dataset = GTZANDataset(metadata_path, audio_dir)
print(f"Running GTZAN dataset with lenght: {len(dataset)}")

# Initialize the CLAP model and processor
model_id = "laion/larger_clap_music_and_speech"
processor = ClapProcessor.from_pretrained(model_id)
model = ClapModel.from_pretrained(model_id).to(device)

# Prepare Datasets
train_len = int(0.8 * len(dataset))
test_len = int(len(dataset) - train_len)

train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

# for audio, label in train_dataset:
#     print(audio.shape, label)

### TRAINING FOR FINE TUNNING
# Set the model to training mode
model.train()
# Freeze the pre-trained model parameters
for param in model.parameters():
    param.requires_grad = False

classes = dataset.classes
class_to_index = {class_name: index for index, class_name in enumerate(classes)}
# Add a new linear layer for classification
num_classes = len(classes)  # Number of classes in your dataset
classifier = nn.Linear(512, num_classes).to(device) # 512 is the number of features in the CLAP model

# Unfreeze the parameters of the new layer
for param in classifier.parameters():
    param.requires_grad = True
# Initialize the optimizer
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# Define a loss function
criterion = nn.CrossEntropyLoss()


# Set the number of epochs
num_epochs = 2
len_dataset = (len(train_dataset))
batch_size = 1

# Train the model
for epoch in range(num_epochs):
    for i in tqdm(range(0, len_dataset, batch_size)):
        i_end = min(i + batch_size, len_dataset)
        audios = [dataset[j][0] for j in range(i, i_end)]
        labels = [dataset[j][1] for j in range(i, i_end)]  # Get the labels for the current batch

        inputs_audio = processor(audios=audios, sampling_rate=48000, return_tensors="pt", padding=True)
        inputs_audio = {key: value.to(device) for key, value in inputs_audio.items()}
        audio_features = model.get_audio_features(**inputs_audio)

        # Forward pass
        outputs = classifier(audio_features)
        print(labels)
        # Convert class names to class indices
        labels = [class_to_index[label] for label in labels]

        # Compute loss
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        # Compute loss
        # labels = torch.tensor(labels, dtype=torch.long, device=device)  # Convert labels to a tensor
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


### EVALUATION
# Set the model to evaluation mode
model.eval()

# Initialize a variable to keep track of the total loss
total_loss = 0

# Initialize a variable to keep track of the total number of correct predictions
total_correct = 0

len_test_dataset = len(test_dataset)

# Iterate over the test dataset
for i in tqdm(range(0, len_test_dataset, batch_size)):
    i_end = min(i + batch_size, len_test_dataset)
    audios = [test_dataset[j][0] for j in range(i, i_end)]
    labels = [test_dataset[j][1] for j in range(i, i_end)]  # Get the labels for the current batch

    inputs_audio = processor(audios=audios, sampling_rate=48000, return_tensors="pt", padding=True)
    inputs_audio = {key: value.to(device) for key, value in inputs_audio.items()}
    audio_features = model.get_audio_features(**inputs_audio)

    # Forward pass
    with torch.no_grad():
        outputs = classifier(audio_features)

    # Convert class names to class indices
    labels = [class_to_index[label] for label in labels]

    # Compute loss
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    loss = criterion(outputs, labels_tensor)
    total_loss += loss.item()

    # Get the predicted labels
    _, predicted = torch.max(outputs, 1)

    # Compute the number of correct predictions
    correct = (predicted == labels_tensor).sum().item()
    total_correct += correct

# Compute the average loss and accuracy
avg_loss = total_loss / len_test_dataset
accuracy = total_correct / len_test_dataset

print(f"Average loss: {avg_loss}, Accuracy: {accuracy}")

