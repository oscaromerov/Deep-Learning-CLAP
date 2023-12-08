from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from audio_datasets.audio_dataset import MusicSentimentDataset
from fine_tunning import FineTuneCLAP
import torch

# Initialize MusicSentimentDataset
# metadata_path = 'path_to_your_metadata'
# audio_dir = 'path_to_your_audio_directory'
metadata_path = "audios/emotions/emotions_music_labels.csv"
audio_dir = "audios/emotions"
dataset = MusicSentimentDataset(metadata_path, audio_dir)
print(f"DATASET: {len(dataset)}")
# for f in range(len(dataset)):
#     print(dataset[f][0].shape)
#     print(dataset[f][1])
#     print(dataset.filenames[f])



from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Prepare your data and labels
X = np.array([dataset[i][0] for i in range(len(dataset))])
y = np.array([dataset[i][1] for i in range(len(dataset))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1, activation='relu')

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
predictions = mlp.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)
# # dataset = MusicSentimentDataset(audio_dir)
# print(f"DATASET: {len(dataset)}")
#
# # Split the dataset into training and testing sets
# train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
# train_dataset = Subset(dataset, train_indices)
# test_dataset = Subset(dataset, test_indices)

# Create DataLoaders for the training and testing sets
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# print(len(train_dataset))
# print(len(test_dataset))
# print(len(train_dataloader))
# # epochs = 10
# # for epoch in range(epochs):
# #     for audio, label in train_dataloader:
# #         print(audio.shape)
# #         print(label)
# #         break
#
# # Initialize FineTuneCLAP
# model_id = "laion/clap-htsat-fused"
# num_classes = len(dataset.classes)  # replace with the number of classes in your task
# model = FineTuneCLAP(metadata_path, audio_dir, 'MusicSentiment', model_id, num_classes)
# print(num_classes)
#
# # Fine-tune the model
# epochs = 2
# learning_rate = 1e-4
# model.fine_tune(train_dataloader, epochs, learning_rate)
#
# # Save the fine-tuned model
# model_path = 'fine_tuned_clap_model.pth'
# model.save_model(model_path)
#
# # Load the fine-tuned model
# model.load_model(model_path)

# # Evaluate the model
# for audios, labels in test_dataloader:
#     logits = model(audios, labels)
#     predictions = torch.argmax(logits, dim=1)
#     accuracy = (predictions == labels).float().mean()
#     print(f"Accuracy: {accuracy.item()}")