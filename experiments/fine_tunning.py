import torch
from audio_classification import AudioClassifier
import torch.nn as nn


class FineTuneCLAP(AudioClassifier):
    def __init__(self, metadata_path, audio_dir, dataset_class, model_id, num_classes):
        super().__init__(metadata_path, audio_dir, dataset_class, model_id)

        # Freeze the pre-trained model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Add a new MLP for classification
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, audios, texts):
        # Process audios and texts with the processor
        inputs_audio = self.processor(audios=audios, sampling_rate=48000, return_tensors="pt", padding=True)
        inputs_text = self.processor(text=texts, return_tensors="pt", padding=True)

        # Get the audio and text features from the pre-trained model
        audio_features = self.model.get_audio_features(**inputs_audio)
        text_features = self.model.get_text_features(**inputs_text)

        # Concatenate the audio and text features
        features = torch.cat((audio_features, text_features), dim=1)

        # Pass the features through the classifier to get the logits
        logits = self.classifier(features)

        return logits

    def fine_tune(self, train_dataloader, epochs, learning_rate):
        # Prepare the optimizer
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)

        # Specify the loss function
        loss_fn = nn.CrossEntropyLoss()

        # Train the model
        for epoch in range(epochs):
            for audios, labels in train_dataloader:
                # Forward pass
                # logits = self(audios, labels)
                logits = self.forward(audios, labels)

                # Compute loss
                loss = loss_fn(logits, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

# Load the fine-tuned model
model.load_state_dict(torch.load('fine_tuned_clap_model.pth'))

# Create an instance of your AudioClassifier with the fine-tuned model
audio_classifier = AudioClassifier(metadata_path, audio_dir, dataset_class, model)

# Evaluate the model
good_predictions, bad_predictions = audio_classifier.predict_and_evaluate(batch_size=1000, len_dataset=len(dataset))

# Print the good and bad predictions
print("Good predictions:", good_predictions)
print("Bad predictions:", bad_predictions)