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
import librosa

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

    def train(self, classifier, num_epochs: int = 10, batch_size: int = 32, lr: float = 0.01,
              name_of_model_to_save: str = 'fine_tuned_clap_model'):
        # Data preparation
        train_len = int(0.8 * len(self.dataset))
        test_len = len(self.dataset) - train_len

        # Split indices into train and test
        self.train_indices, self.test_indices = train_test_split(
            range(len(self.dataset)), test_size=test_len
        )
        self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_indices)
        self.test_dataset = torch.utils.data.Subset(self.dataset, self.test_indices)
        random.shuffle(self.train_indices)

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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total training time: {elapsed_time} seconds")
        torch.save(classifier.state_dict(), f"experiments/fine_tune_models/{name_of_model_to_save}.pth")
        return train_losses

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
            true_labels.extend([self.index_to_class[label] for label in labels])
            pred_labels.extend([self.index_to_class[pred] for pred in predicted.tolist()])
        avg_loss = total_loss / len_test_dataset
        accuracy = total_correct / len_test_dataset
        print(f"During Evaluation in unseen data metrics are -> Average loss: {avg_loss}, Accuracy: {accuracy}")
        return avg_loss, accuracy, true_labels, pred_labels

    def plot_loss(self, train_losses):
        plt.plot(train_losses, label='Training loss')
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
        
    def predict(self, audio_path, classifier):
        self.model.eval()
        waveform, sr = librosa.load(audio_path, sr=None)
        inputs_audio = self.processor(audios=waveform, sampling_rate=48000, return_tensors="pt", padding=True)
        inputs_audio = {key: value.to(self.device) for key, value in inputs_audio.items()}
        audio_features = self.model.get_audio_features(**inputs_audio)
        with torch.no_grad(): #don't calculate gradients as we are not updating the weights during prediction
            outputs = classifier(audio_features)
            _, predicted = torch.max(outputs, 1) #get the class with the highest probability
            predicted_class = self.index_to_class[predicted.item()] # convert the class index to class name
        print(f"The predicted class for the audio file {audio_path} is {predicted_class}.")