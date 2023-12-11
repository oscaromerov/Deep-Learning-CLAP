from torch.utils.data import DataLoader
from audio_datasets.audio_dataset import GTZANDataset
from transformers import ClapModel, ClapProcessor
import torch
from tqdm.auto import tqdm
from torch.utils.data import random_split
import torch.nn as nn
import random
import time
# Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device available:{device}")



# Initialize the CLAP model and processor
model_id = "laion/larger_clap_music_and_speech"
processor = ClapProcessor.from_pretrained(model_id)
model = ClapModel.from_pretrained(model_id).to(device)

# Initialize GTZANDataset
metadata_path = "audios/GTZAN/features_30_sec.csv"
audio_dir = "audios/GTZAN/genres_original"
dataset_class = 'GTZAN'
dataset = GTZANDataset(metadata_path, audio_dir, processor)
print(f"Running GTZAN dataset with lenght: {len(dataset)}")

# Prepare Datasets
train_len = int(0.8 * len(dataset))
test_len = int(len(dataset) - train_len)
print(f"Train len: {train_len}, Test len: {test_len}")


## Dataset Preparation
from sklearn.model_selection import train_test_split

# Get the labels from the dataset
labels = [item[1] for item in dataset]

# Perform a stratified split
train_indices, test_indices = train_test_split(
    range(len(dataset)), test_size=test_len, stratify=labels
)

# Create the train and test datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# for audio, label in train_dataset:
#     print(audio.shape, label)

# Create a list of indices and shuffle it
# indices = list(range(len_dataset))
# random.shuffle(train_indices)

# Decide what proportion of the training dataset you want to use for validation
valid_ratio = 0.1  # Use 10% of the training dataset for validation
###### BEFORE CHANGES #######
# Calculate the number of samples to use for validation
# valid_len = int(valid_ratio * len(train_dataset))
# train_len = int(len(train_dataset) - valid_len)
# train_labels = [item[1] for item in train_dataset]
# print(f"Train len: {train_len}, Valid len: {valid_len}")
#
# # Perform a stratified split
# train_indices, valid_indices = train_test_split(
#     range(len(train_dataset)), test_size=valid_len, stratify=train_labels
# )
#
# # Create the new train and validation datasets
# train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
# valid_dataset = torch.utils.data.Subset(train_dataset, valid_indices)

####################################

# random.shuffle(train_indices)
# random.shuffle(valid_indices)

from torch.utils.data import SubsetRandomSampler
import numpy as np

# obtain training indices that will be used for validation
indices = list(range(len(train_dataset)))
np.random.shuffle(indices)
split = int(np.floor(valid_ratio * len(train_dataset)))
train_index, valid_index = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)

# Set the number of epochs
num_epochs = 10
# len_dataset = (len(train_dataset))
batch_size = 32

# Create DataLoaders for training, validation, and testing
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = dataset.classes
num_classes = len(classes)  # Number of classes in your dataset
classifier = nn.Linear(512, num_classes).to(device)

# Set the model to training mode
start_time = time.time()
model.train()
# Freeze the pre-trained model parameters
for param in model.parameters():
    param.requires_grad = False

# classes = dataset.classes
class_to_index = {class_name: index for index, class_name in enumerate(classes)}
index_to_class = {index: class_name for class_name, index in class_to_index.items()}
# Initialize the optimizer
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# Define a loss function
criterion = nn.CrossEntropyLoss()
# Train the model
train_losses = []
valid_losses = []
# train_loss = 0
best_valid_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = 0
    valid_loss = 0
    # for audios, labels in tqdm(train_loader):
    for i, (audios, labels) in enumerate(tqdm(train_loader)):
        batch_loader = len(audios['input_features'])
        batch_indices = train_indices[i * batch_loader: (i * batch_loader) + batch_loader]
        actual_labels = [dataset.labels[index] for index in batch_indices]
        # print(f"Batch Size: {batch_loader}")
        # print(f"Batch indices: {batch_indices}")
        # print(f"Actual labels: {actual_labels}")

        # Move audios and labels to the device
        audios = audios.to(device)
        labels = labels.to(device)
        # labels = labels['input_ids'].squeeze().long().to(device)
        # labels = labels['input_ids'].squeeze().to(device)[:, 0]
        # print("Loading labels")
        # labels = labels['input_ids'].squeeze(1)
        # text = {key: value.to(device) for key, value in labels.items()}
        labels['input_ids'] = labels['input_ids'].squeeze(1)
        text_features = model.get_text_features(**labels)
        # print("Text feautes loaded")
        # audios = audios.squeeze(1)
        audios['input_features'] = audios['input_features'].squeeze(1)
        audio_features = model.get_audio_features(**audios)

        # Forward pass
        outputs = classifier(audio_features)
        outputs_text = classifier(text_features)

        # Convert class names to class indices
        # labels = [class_to_index[label] for label in labels]
        # Convert class names to class indices
        # labels = [class_to_index[label['input_ids'].item()] for label in labels]

        # Compute loss
        # loss = criterion(outputs, labels)
        # print("Computing loss")
        # ground_truth = torch.arange(0, num_classes).long().to(device)
        # loss = criterion(outputs, ground_truth) + criterion(outputs_text, ground_truth)
        ground_truth = torch.tensor([class_to_index[label] for label in actual_labels], device=device)
        # ground_truth = torch.arange(len(audios)).long().to(device)
        # print(f"grount truth: {ground_truth}")
        loss = (criterion(outputs, ground_truth) + criterion(outputs_text, ground_truth))/2
        # loss = (criterion(outputs, labels['input_ids']) + criterion(outputs_text, labels['input_ids'])) / 2
        # print("Loss computed")

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
        # print(f"Epoch [{epoch + 1}/{num_epochs}], TrainLoss: {train_loss}")

    # Print training loss
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}")
    # Validate the model
    print("Running Validation during training")
    model.eval()
    # valid_loss = 0
    with torch.no_grad():
        # for audios, labels in tqdm(valid_loader):
        for i, (audios, labels) in enumerate(tqdm(valid_loader)):
            batch_loader = len(audios['input_features'])
            batch_indices = train_indices[i * batch_loader: (i * batch_loader) + batch_loader]
            actual_labels = [dataset.labels[index] for index in batch_indices]
            # print(f"Batch Size: {batch_loader}")
            # print(f"Batch indices: {batch_indices}")
            # print(f"Actual labels: {actual_labels}")
            audios = audios.to(device)
            labels = labels.to(device)

            # labels = labels['input_ids'].squeeze().to(device)[:, 0]
            # print("Loading labels")
            labels['input_ids'] = labels['input_ids'].squeeze(1)
            text_features = model.get_text_features(**labels)
            # print("Text feautes loaded")

            audios['input_features'] = audios['input_features'].squeeze(1)
            audio_features = model.get_audio_features(**audios)
            # Forward pass
            outputs = classifier(audio_features)
            outputs_text = classifier(text_features)

            ground_truth = torch.tensor([class_to_index[label] for label in actual_labels], device=device)
            print(f"grount truth: {ground_truth}")
            loss = (criterion(outputs, ground_truth) + criterion(outputs_text, ground_truth)) / 2
            # loss = criterion(outputs, labels)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)
    # Print training and validation loss for this epoch
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}")
    # Save the model if validation loss has decreased
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(classifier.state_dict(), 'experiments/fine_tune_models/DEMO_classifier.pth')

# Evaluation
print("Starting evaluation")
model.eval()
total_loss = 0
total_correct = 0
with torch.no_grad():
    # for audios, labels in tqdm(test_loader):
    for i, (audios, labels) in enumerate(tqdm(test_loader)):
        batch_loader = len(audios['input_features'])
        batch_indices = train_indices[i * batch_loader: (i * batch_loader) + batch_loader]
        actual_labels = [dataset.labels[index] for index in batch_indices]
        # print(f"Batch Size: {batch_loader}")
        # print(f"Batch indices: {batch_indices}")
        # print(f"Actual labels: {actual_labels}")
        # Move audios and labels to the device
        audios = audios.to(device)
        labels = labels.to(device)

        labels['input_ids'] = labels['input_ids'].squeeze(1)
        text_features = model.get_text_features(**labels)
        # print("Text feautes loaded")
        audios['input_features'] = audios['input_features'].squeeze(1)
        audio_features = model.get_audio_features(**audios)


        # Forward pass
        outputs = classifier(audio_features)
        outputs_text = classifier(text_features)
        # loss = criterion(outputs, labels)
        ground_truth = torch.tensor([class_to_index[label] for label in actual_labels], device=device)
        # ground_truth = torch.arange(len(audios)).long().to(device)
        print(f"grount truth: {ground_truth}")
        loss = (criterion(outputs, ground_truth) + criterion(outputs_text, ground_truth)) / 2
        total_loss += loss.item()

        # Get the predicted labels
        _, predicted = torch.max(outputs, 1)

        # Compute the number of correct predictions
        # correct = (predicted == labels).sum().item()
        correct = (predicted == ground_truth).sum().item()
        total_correct += correct
        # Convert the predicted and actual labels to lists
        predicted_labels = [index_to_class[index] for index in predicted.tolist()]
        actual_labels = [index_to_class[index] for index in ground_truth.tolist()]

        for predicted_label, actual_label in zip(predicted_labels, actual_labels):
            print(f"Predicted: {predicted_label}, Actual: {actual_label}")

        # predicted_labels = [index_to_class[index] for index in predicted.tolist()]
        # actual_labels = [index_to_class[index] for index in labels.tolist()]
        # # Print the predicted and actual labels
        # for predicted_label, actual_label in zip(predicted_labels, actual_labels):
        #     print(f"Predicted: {predicted_label}, Actual: {actual_label}")

# # Compute the average loss and accuracy
# avg_loss = total_loss / len(test_loader)
# accuracy = total_correct / len(test_loader.dataset)
#
# print(f"Average loss: {avg_loss}, Accuracy: {accuracy}")
#
# import matplotlib.pyplot as plt
#
# plt.plot(train_losses, label='Training loss')
# plt.plot(valid_losses, label='Validation loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()