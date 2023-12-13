from experiments.audio_classification import AudioClassifier
from experiments.text_to_audio import Text2Audio
from experiments.fine_tune_audio_classification import FineTuneAudioClassifier
import utils
import torch.nn as nn


def main():

    config = utils.load_json_config("main.json")

    print("""Bonjour! Welcome to the CLAP demo.
          There are 3 experiments you can run, please select one:
          """)
    for i, experiment in enumerate(config['experiments'], 1):
        print(f"{i}. {experiment}")
    choice = int(input("Enter the number of your choice: ")) - 1
    if choice not in range(len(config['experiments'])):
        raise ValueError("Invalid choice for experiment.")
    experiment = config['experiments'][choice]

    print("Please select the dataset you want to use:")
    for i, dataset in enumerate(config['datasets'], 1):
        print(f"{i}. {dataset}")
    choice = int(input("Enter the number of your choice: ")) - 1
    if choice not in range(len(config['experiments'])):
        raise ValueError("Invalid choice for experiment.")
    dataset = config['datasets'][choice]

    print("Please select the model you want to use, keep in mind the dataset you chose before:")
    for i, model in enumerate(config['models'], 1):
        print(f"{i}. {model}")
    choice = int(input("Enter the number of your choice: ")) - 1
    if choice not in range(len(config['models'])):
        raise ValueError("Invalid choice for model.")
    model = config['models'][choice]

    experiment_config = config['configurations'][dataset]
    experiment_config['dataset_class'] = dataset
    experiment_config['model_id'] = model

    if experiment == 'Zero-Shot Audio Classification':
        audio_classifier = AudioClassifier(**experiment_config)
        good_predictions, bad_predictions, accuracy, true_labels, pred_labels = audio_classifier.predict_and_evaluate(
            batch_size=32)
        audio_classifier.plot_predictions_by_class(good_predictions, bad_predictions)
        audio_classifier.get_confusion_matrix(true_labels, pred_labels)


    elif experiment == 'Text to Audio Retrieval':
        print("Please enter the text query you want to use to retrieve an audio")
        text_query = str(input("Enter the text query: "))
        text2audio = Text2Audio(text_query=text_query, **experiment_config)
        audio_embeddings = text2audio.get_audio_embeddings()
        similarities = text2audio.get_similarities(audio_embeddings)
        top_indices = text2audio.get_top_indices(similarities, k=5)
        idx = top_indices[0]
        text2audio.dataset.display_waveform(idx)
        text2audio.dataset.display_spectrogram(idx)
        text2audio.dataset.play_audio(idx)

    elif experiment == 'Fine-tune Audio Classification':
        fine_tune_audio_classifier = FineTuneAudioClassifier(**experiment_config)
        classifier = nn.Linear(512, fine_tune_audio_classifier.num_classes)
        train_losses = fine_tune_audio_classifier.train(classifier, num_epochs=3)
        avg_loss, accuracy, true_labels, pred_labels = fine_tune_audio_classifier.evaluate(classifier)
        fine_tune_audio_classifier.plot_loss(train_losses)
        fine_tune_audio_classifier.plot_confusion_matrix(true_labels, pred_labels)

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
