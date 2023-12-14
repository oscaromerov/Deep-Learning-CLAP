from audio_datasets.audio_dataset import ESC50Dataset, GTZANDataset, MusicSentimentDataset
from transformers import ClapModel, ClapProcessor
import torch
from tqdm.auto import tqdm
from torch.utils.data import random_split
import torch.nn as nn
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd


class FineTuneAudioClassifier:
    def __init__(self, metadata_path, audio_dir, dataset_class, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if dataset_class == 'ESC50':
            self.dataset = ESC50Dataset(metadata_path, audio_dir)
        elif dataset_class == 'GTZAN' or dataset_class == "MusicGen":
            self.dataset = GTZANDataset(metadata_path, audio_dir)
        elif dataset_class == 'MusicSentiment':
            self.dataset = MusicSentimentDataset(metadata_path, audio_dir)
        else:
            raise ValueError("Invalid dataset choice.")
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(self.device)
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
        self.class_to_index = {class_name: index for index, class_name in enumerate(self.classes)}
        self.index_to_class = {index: class_name for class_name, index in self.class_to_index.items()}
        print(f"Running {dataset_class} dataset with lenght: {len(self.dataset)} classifying this number of clases: {self.classes}")

    def train(self, classifier, num_epochs: int = 10, batch_size: int = 32, lr: float = 0.01):
        # Data preparation
        train_len = int(0.8 * len(self.dataset))
        test_len = len(self.dataset) - train_len

        # Split indices into train and test
        self.train_indices, self.test_indices = train_test_split(
            range(len(self.dataset)), test_size=test_len, random_state=46, shuffle=True)

        # Split train indices into train and validation
        valid_ratio = 0.1
        valid_len = int(valid_ratio * train_len)
        self.train_indices, self.valid_indices = train_test_split(
            self.train_indices, test_size=valid_len, random_state=46, shuffle=True
        )

        # Create subsets
        self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_indices)
        self.valid_dataset = torch.utils.data.Subset(self.dataset, self.valid_indices)
        self.test_dataset = torch.utils.data.Subset(self.dataset, self.test_indices)

        print("Number of samples in train set:", len(self.train_dataset))
        print("Number of samples in validation set:", len(self.valid_dataset))
        print("Number of samples in test set:", len(self.test_dataset))


        # Training
        start_time = time.time()
        self.model.train()
        # Freeze the pre-trained model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        classifier = classifier.to(self.device)
        # Unfreeze the parameters of the new layers and set them to trainable
        for param in classifier.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = 0
            test_classes = set([self.test_dataset[j][1] for j in range(test_len)])
            print("Classes in the test set:", test_classes)
            train_classes = set([self.train_dataset[j][1] for j in range(train_len)])
            print("Classes in the test set:", train_classes)
            for i in tqdm(range(0, len(self.train_dataset), batch_size)):
                i_end = min(i + batch_size, len(self.train_dataset))
                audios = [self.dataset[self.train_indices[j]][0] for j in range(i, i_end)]
                labels = [self.dataset[self.train_indices[j]][1] for j in range(i, i_end)]
                inputs_audio = self.processor(audios=audios, sampling_rate=48000, return_tensors="pt", padding=True)
                inputs_audio = {key: value.to(self.device) for key, value in inputs_audio.items()}
                audio_features = self.model.get_audio_features(**inputs_audio)
                outputs = classifier(audio_features)
                labels = [self.class_to_index[label] for label in labels]
                labels = torch.tensor(labels, dtype=torch.long, device=self.device)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(self.train_dataset)
            train_losses.append(avg_train_loss)
            self.model.eval()
            valid_loss = 0
            with torch.no_grad():
                for i in tqdm(range(0, len(self.valid_dataset), batch_size)):
                    i_end = min(i + batch_size, len(self.valid_dataset))
                    audios = [self.dataset[self.valid_indices[j]][0] for j in range(i, i_end)]
                    labels = [self.dataset[self.valid_indices[j]][1] for j in range(i, i_end)]
                    inputs_audio = self.processor(audios=audios, sampling_rate=48000, return_tensors="pt", padding=True)
                    inputs_audio = {key: value.to(self.device) for key, value in inputs_audio.items()}
                    audio_features = self.model.get_audio_features(**inputs_audio)
                    outputs = classifier(audio_features)
                    labels = [self.class_to_index[label] for label in labels]
                    labels = torch.tensor(labels, dtype=torch.long, device=self.device)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
            avg_valid_loss = valid_loss / len(self.valid_dataset)
            valid_losses.append(avg_valid_loss)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(classifier.state_dict(), 'experiments/fine_tune_models/classifier.pth')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total training time: {elapsed_time} seconds")
        return train_losses, valid_losses

    def evaluate(self, classifier, batch_size=32):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        len_test_dataset = len(self.test_dataset)
        criterion = nn.CrossEntropyLoss()
        true_labels = []
        pred_labels = []
        for i in tqdm(range(0, len_test_dataset, batch_size)):
            i_end = min(i + batch_size, len_test_dataset)
            audios = [self.test_dataset[j][0] for j in range(i, i_end)]
            labels = [self.test_dataset[j][1] for j in range(i, i_end)]
            inputs_audio = self.processor(audios=audios, sampling_rate=48000, return_tensors="pt", padding=True)
            inputs_audio = {key: value.to(self.device) for key, value in inputs_audio.items()}
            audio_features = self.model.get_audio_features(**inputs_audio)
            with torch.no_grad():
                outputs = classifier(audio_features)
            labels = [self.class_to_index[label] for label in labels]
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
            loss = criterion(outputs, labels_tensor)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels_tensor).sum().item()
            total_correct += correct
            # true_labels.extend(labels)
            # pred_labels.extend(predicted.tolist())
            true_labels.extend([self.index_to_class[label] for label in labels])
            pred_labels.extend([self.index_to_class[pred] for pred in predicted.tolist()])
        avg_loss = total_loss / len_test_dataset
        accuracy = total_correct / len_test_dataset
        print(f"During Evaluation in unseen data metrics are -> Average loss: {avg_loss}, Accuracy: {accuracy}")
        return avg_loss, accuracy, true_labels, pred_labels

    def plot_loss(self, train_losses, valid_losses):
        plt.plot(train_losses, label='Training loss')
        plt.plot(valid_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, true_labels, pred_labels):
        cm = confusion_matrix(true_labels, pred_labels, labels=self.classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical')
        plt.show()


# Initialize the FineTuneAudioClassifier
# metadata_path = "audios/GTZAN/features_30_sec.csv"
# audio_dir = "audios/GTZAN/genres_original"
# dataset_class = 'GTZAN'
# model_id = "laion/larger_clap_music_and_speech"
# fine_tune_audio_classifier = FineTuneAudioClassifier(metadata_path, audio_dir, dataset_class, model_id)

# Prepare datasets
# fine_tune_audio_classifier.prepare_datasets()

# Define the classifier model
# classifier = nn.Linear(512, fine_tune_audio_classifier.num_classes)
#
# # Train the model
# train_losses, valid_losses = fine_tune_audio_classifier.train(classifier, num_epochs=2)
#
# # Evaluate the model
# avg_loss, accuracy, true_labels, pred_labels = fine_tune_audio_classifier.evaluate(classifier)
#
# # train_losses, valid_losses, elapsed_time = fine_tune_audio_classifier.train(classifier)
# fine_tune_audio_classifier.plot_loss(train_losses, valid_losses)
# fine_tune_audio_classifier.plot_confusion_matrix(true_labels, pred_labels)
# good_predictions, bad_predictions, acc, true_labels, pred_labels = fine_tune_audio_classifier.predict_and_evaluate(batch_size=32)
# fine_tune_audio_classifier.plot_classified_labels(good_predictions, bad_predictions)