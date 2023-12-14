from audio_datasets.audio_dataset import ESC50Dataset, GTZANDataset, MusicSentimentDataset
from transformers import ClapModel, ClapProcessor
import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import utils

class AudioClassifier:
    text_label_augmentation = utils.load_json_config("text_label_augmentation.json")
    def __init__(self, metadata_path: str, audio_dir: str, dataset_class: str,
                 model_id: str, text_augmentation: bool = False):
        # Set Device where model is going to run
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
       # Define Dataset to use
        self.dataset_class = dataset_class
        if self.dataset_class == 'ESC50':
            self.dataset = ESC50Dataset(metadata_path, audio_dir)
        elif self.dataset_class == 'GTZAN' or self.dataset_class == "MusicGen":
            self.dataset = GTZANDataset(metadata_path, audio_dir)
        elif self.dataset_class == 'MusicSentiment':
            self.dataset = MusicSentimentDataset(metadata_path, audio_dir)
        else:
            raise ValueError("Invalid dataset choice.")

        self.classes = self.dataset.classes
        self.text_augmentation = text_augmentation
        if self.text_augmentation:
            self.classes_sentences = [self.text_label_augmentation['datasets'][self.dataset_class]['augmented_labels'][c]
                                      for c in self.classes]
        else:
            self.classes_sentences = [self.text_label_augmentation['datasets'][self.dataset_class]['simple_labels'][c]
                                      for c in self.classes]
        # Load model and processor
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(self.device)
        self.inputs_text = self.processor(text=self.classes_sentences, return_tensors="pt", padding=True)
        self.inputs_text = {key: value.to(self.device) for key, value in self.inputs_text.items()}
        with torch.inference_mode():
            self.outputs_text = self.model.get_text_features(**self.inputs_text)
        self.text_emb = self.outputs_text.detach().cpu().numpy()
        print(f"Running model: {model_id} over dataset: {self.dataset_class} with length: {len(self.dataset)}")

    def predict_and_evaluate(self, batch_size: int = 32, len_dataset: int = None):
        if len_dataset is None:
            len_dataset = len(self.dataset)
        else:
            len_dataset = min(len_dataset, len(self.dataset))
        predictions = []
        true_labels = []
        pred_labels = []
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
            predictions.extend(np.argmax(scores, axis=1))
        true_predictions = []
        for i in range(len_dataset):
            waveform, label = self.dataset[i]
            if self.classes[predictions[i]] == label:
                true_labels.append(label)
                pred_labels.append(self.classes[predictions[i]])
                true_predictions.append(1)
                good_predictions[label] += 1
            else:
                true_labels.append(label)
                pred_labels.append(self.classes[predictions[i]])
                true_predictions.append(0)
                bad_predictions[label] += 1
        accuracy = sum(true_predictions) / len(true_predictions)
        print(f"After running zero-shot classification over {len_dataset} samples on {self.dataset_class} dataset, the accuracy is: {accuracy}")
        return good_predictions, bad_predictions, accuracy, true_labels, pred_labels

    def plot_predictions_by_class(self, good_predictions, bad_predictions):
        labels = sorted(self.classes)
        good_counts = [good_predictions[label] for label in labels]
        bad_counts = [bad_predictions[label] for label in labels]
        x = np.arange(len(labels))
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        ax.bar(x, good_counts, width, label='Good')
        ax.bar(x, bad_counts, width, bottom=good_counts, label='Bad')
        ax.set_xlabel('Labels Classes')
        ax.set_ylabel('Counts')
        ax.set_title('Good and Bad Predictions by Label Class')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        plt.xticks(rotation=90)  # Rotate x-axis labels
        ax.legend()
        fig.tight_layout()
        plt.show()


    def get_confusion_matrix(self, true_labels, pred_labels):
        cm = confusion_matrix(true_labels, pred_labels, labels=self.classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical')
        plt.show()