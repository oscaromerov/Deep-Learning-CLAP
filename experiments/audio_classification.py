from audio_datasets.audio_dataset import ESC50Dataset, GTZANDataset, UrbanSound8kDataset, MusicSentimentDataset
from transformers import ClapModel, ClapProcessor
import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from text_label_augmentation import augmentations_esc
import utils
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity


class AudioClassifier:
    def __init__(self, metadata_path, audio_dir, dataset_class, model_id):
        if dataset_class == 'ESC50':
            self.dataset = ESC50Dataset(metadata_path, audio_dir)
        elif dataset_class == 'GTZAN':
            self.dataset = GTZANDataset(metadata_path, audio_dir)
        elif dataset_class == 'UrbanSound8k':
            self.dataset = UrbanSound8kDataset(metadata_path, audio_dir)
        elif dataset_class == 'MusicSentiment':
            self.dataset = MusicSentimentDataset(metadata_path, audio_dir)
        else:
            raise ValueError("Invalid dataset choice.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes = self.dataset.classes
        text_label_augmentation = utils.load_json_config("text_label_augmentation.json")
        if dataset_class == 'MusicSentiment':
            self.classes_sentences = [text_label_augmentation['datasets'][dataset_class]['augmented_labels'][c]
            for c in self.classes]
            # self.classes_sentences = [f"This audio is a song which makes me feel {c}" for c in self.classes]
        elif dataset_class == 'ESC50' or dataset_class == 'UrbanSound8k':
            self.classes_sentences = [text_label_augmentation['datasets'][dataset_class]['augmented_labels'][c]
                                      for c in self.classes]
            # self.classes_sentences = [f"This is a sound of {c}" for c in self.classes]
        elif dataset_class == 'GTZAN':
            # self.classes_sentences = [text_label_augmentation['datasets'][dataset_class]['augmented_labels'][c]
            #                           for c in self.classes]
            self.classes_sentences = [f"This audio is a {c} song" for c in self.classes]
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(self.device)
        self.inputs_text = self.processor(text=self.classes_sentences, return_tensors="pt", padding=True)
        self.inputs_text = {key: value.to(self.device) for key, value in self.inputs_text.items()}
        with torch.inference_mode():
            self.outputs_text = self.model.get_text_features(**self.inputs_text)
        self.text_emb = self.outputs_text.detach().cpu().numpy()

        print(f"The Dataset len is: {len(self.dataset)}")
        print(f"Ruuning model: {model_id} over dataset: {dataset_class}")
        print(f"Classes sentences: {self.classes_sentences}")

    def predict_and_evaluate(self, batch_size, len_dataset: int = None):
        if len_dataset is None:
            len_dataset = len(self.dataset)
        else:
            len_dataset = min(len_dataset, len(self.dataset))
        predictions = []
        good_predictions = {class_name: 0 for class_name in self.classes}
        bad_predictions = {class_name: 0 for class_name in self.classes}
        for i in tqdm(range(0, len_dataset, batch_size)):
            i_end = min(i + batch_size, len_dataset)
            audios = [self.dataset[j][0] for j in range(i, i_end)]
            inputs_audio = self.processor(audios=audios, sampling_rate=48000, return_tensors="pt", padding=True)
            inputs_audio = {key: value.to(self.device) for key, value in inputs_audio.items()}
            with torch.inference_mode():
                outputs_audio = self.model.get_audio_features(**inputs_audio)
            audio_emb = outputs_audio.detach().cpu().numpy()
            assert audio_emb.shape[1] == self.text_emb.shape[1], "Audio and text embeddings must have the same dimension"
            scores = np.dot(audio_emb, self.text_emb.T)
            # scores = cosine_similarity(audio_emb, self.text_emb.T)
            predictions.extend(np.argmax(scores, axis=1))

        true_predictions = []
        for i in range(len_dataset):
            waveform, label = self.dataset[i]
            print(
            f"True label: {label}, Pred label: {self.classes[predictions[i]]}, filename: {self.dataset.filenames[i]}")
            if self.classes[predictions[i]] == label:
                true_predictions.append(1)
                good_predictions[label] += 1
            else:
                true_predictions.append(0)
                bad_predictions[label] += 1

        accuracy = sum(true_predictions) / len(true_predictions)
        print(f"Accuracy: {accuracy}")
        return good_predictions, bad_predictions

    def plot_predictions_by_class(self, good_predictions, bad_predictions):
        # Calculate accuracy for each class
        accuracies = {label: good_predictions[label] / (good_predictions[label] + bad_predictions[label])
                      for label in good_predictions.keys() if good_predictions[label] > 0 or bad_predictions[label] > 0}
        sorted_labels = sorted(accuracies, key=accuracies.get)

        # Select the 3 classes with the lowest accuracy
        worst_labels = sorted_labels[:3]

        # Get good and bad counts for the worst classes
        worst_good_counts = [good_predictions[label] for label in worst_labels]
        worst_bad_counts = [bad_predictions[label] for label in worst_labels]

        x = np.arange(len(worst_labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, worst_good_counts, width, label='Good')
        rects2 = ax.bar(x + width / 2, worst_bad_counts, width, label='Bad')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Classes')
        ax.set_ylabel('Counts')
        ax.set_title('Good and Bad Predictions by Class for the 3 Worst Classes')
        ax.set_xticks(x)
        ax.set_xticklabels(worst_labels)
        ax.legend()
        fig.tight_layout()
        plt.show()

    # def fine_tune_mlp(self, hidden_layer_sizes=(256,), max_iter=300, alpha=1e-4,
    #                   solver='sgd', verbose=10, random_state=1, learning_rate_init=.1, activation='relu'):
    #     # Prepare your data and labels
    #     X = np.array([self.dataset[i][0] for i in range(len(self.dataset))])
    #     y = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
    #
    #     # Split the data into training and testing sets
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    #     # Initialize the MLPClassifier
    #     mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha,
    #                         solver=solver, verbose=verbose, random_state=random_state,
    #                         learning_rate_init=learning_rate_init, activation=activation)
    #
    #     # Train the model
    #     mlp.fit(X_train, y_train)
    #
    #     # Make predictions
    #     predictions = mlp.predict(X_test)
    #
    #     # Calculate accuracy
    #     accuracy = accuracy_score(y_test, predictions)
    #     print("Accuracy: ", accuracy)
    #
    #     return mlp, accuracy

    def split_dataset(self, train_size=0.7, val_size=0.15, batch_size=32):
        # Calculate the lengths of the splits
        train_len = int(train_size * len(self.dataset))
        val_len = int(val_size * len(self.dataset))
        test_len = len(self.dataset) - train_len - val_len

        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_len, val_len, test_len])

        # Create DataLoaders for the datasets
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_dataloader, val_dataloader, test_dataloader

    # def fine_tune(self, num_epochs, train_dataloader):
    #     # Freeze the pre-trained model parameters
    #     for param in self.model.parameters():
    #         param.requires_grad = False
    #
    #     # Add a new linear layer for classification
    #     num_classes = len(self.classes)  # Number of classes in your dataset
    #     self.classifier = nn.Linear(512, num_classes) # 512 is the number of features in the CLAP model
    #
    #     # Unfreeze the parameters of the new layer
    #     for param in self.classifier.parameters():
    #         param.requires_grad = True
    #
    #     # Define a loss function
    #     criterion = nn.CrossEntropyLoss()
    #
    #     # Define an optimizer that will only update the parameters of the new layer
    #     optimizer = Adam(self.classifier.parameters(), lr=0.001)
    #
    #     # Train the model
    #     for epoch in range(num_epochs):
    #         # for audios, labels in tqdm(train_dataloader, total=len(train_dataloader)):
    #         for audios, labels in train_dataloader:
    #             # audios = audios.to(self.device)
    #             # labels = labels.to(self.device)
    #             # Compute input features from audios
    #             inputs_audio = self.processor(audios=audios, sampling_rate=48000, return_tensors="pt", padding=True)
    #             inputs_audio = {key: value.to(self.device) for key, value in inputs_audio.items()}
    #
    #             # Forward pass
    #             outputs = self.model(**inputs_audio)
    #             loss = criterion(outputs, labels)
    #
    #             # Backward pass and optimization
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")



    def fine_tune(self, num_epochs, train_dataloader):
        # Prepare your data and labels
        X = []
        y = []
        for audios, labels in train_dataloader:
            [...]

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Initialize the classifier
        clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,)))

        # Train the model
        clf.fit(X, y)

        return clf

    def evaluate(self, test_dataloader):
        # Set the model to evaluation mode
        self.model.eval()

        total_correct = 0
        total_predictions = 0

        # Disable gradient calculation
        with torch.no_grad():
            for audios, labels in test_dataloader:
                # Move the audios and labels to the same device as the model
                audios = audios.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(audios)
                _, predicted = torch.max(outputs.data, 1)

                # Update counters
                total_predictions += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        # Calculate accuracy
        accuracy = total_correct / total_predictions
        print(f"Accuracy: {accuracy}")
        return accuracy


# test MusicSentimentDataset
# metadata_path = "audios/emotions/emotions_music_labels.csv"
# audio_dir = "audios/emotions"
# dataset_class = 'MusicSentiment'
# model_id = "laion/clap-htsat-fused"
# # # model_id ="laion/larger_clap_music"
# # # audio_classifier = AudioClassifier(metadata_path, audio_dir, dataset_class, model_id)
# # # good_predictions, bad_predictions = audio_classifier.predict_and_evaluate(batch_size=2, len_dataset=10)
# # # audio_classifier.plot_predictions_by_class(good_predictions, bad_predictions)
# audio_classifier = AudioClassifier(metadata_path, audio_dir, dataset_class, model_id)
# good_predictions, bad_predictions = audio_classifier.predict_and_evaluate(batch_size=2)
# # Fine-tune the model with an MLP
# mlp, accuracy = audio_classifier.fine_tune_mlp()

#test GTZAN dataset
metadata_path = "audios/GTZAN/features_30_sec.csv"
audio_dir = "audios/GTZAN/genres_original"
dataset_class = 'GTZAN'
model_id = "laion/larger_clap_music_and_speech"

# model_id = "laion/larger_clap_music"
# model_id = "laion/clap-htsat-fused"
audio_classifier = AudioClassifier(metadata_path, audio_dir, dataset_class, model_id)
# good_predictions, bad_predictions = audio_classifier.predict_and_evaluate(batch_size=2)

# Initialize the AudioClassifier
# audio_classifier = AudioClassifier(metadata_path, audio_dir, 'GTZAN', model_id)

# Split the dataset into training, validation, and test sets
train_dataloader, val_dataloader, test_dataloader = audio_classifier.split_dataset()

# Define the number of epochs for fine-tuning
num_epochs = 5  # or any other number

# Fine-tune the model
audio_classifier.fine_tune(num_epochs, train_dataloader)

# Evaluate the fine-tuned model on the test set
accuracy = audio_classifier.evaluate(test_dataloader)

print(f"Accuracy of the fine-tuned model on the test set: {accuracy}")

# test ESC50 dataset
# metadata_path = "audios/ESC-50/ESC-50-master/meta/esc50.csv"
# audio_dir = "audios/ESC-50/ESC-50-master/audio"
# dataset_class = 'ESC50'
# #model_id = "laion/larger_clap_music"
# model_id = "laion/clap-htsat-unfused"
# audio_classifier = AudioClassifier(metadata_path, audio_dir, dataset_class, model_id)
# good_predictions, bad_predictions = audio_classifier.predict_and_evaluate(batch_size=2)

