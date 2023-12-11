# import json
# from tqdm import tqdm
#
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
#
# from transformers import ClapProcessor
# from experiments.audio_classification import AudioClassifier
#
# json_path = 'path to train_data.json'
# audio_path = 'path to training dataset'
#
# with open(json_path, 'r') as f:
#     input_data = []
#     for line in f:
#         obj = json.loads(line)
#         input_data.append(obj)
#
# # Load the CLIP model and processor
# model_id = "laion/clap-htsat-fused"
# metadata_path = 'audios/ESC-50/ESC-50-master/meta/esc50.csv'
# audio_dir = 'audios/ESC-50/ESC-50-master/audio/'
# dataset_class = 'ESC50'
# model = AudioClassifier(metadata_path, audio_dir, dataset_class, model_id)
# processor = ClapProcessor.from_pretrained(model_id)
#
# # Choose computation device
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
#
#
# # Define a custom dataset
# class audio_title_dataset():
#     def __init__(self, list_audio_path, list_txt):
#         # Initialize audio paths and corresponding texts
#         self.audio_path = list_audio_path
#         # Tokenize text using CLAP's processor
#         self.title = processor(text=list_txt, return_tensors="pt", padding=True)["input_ids"]
#
#     def __len__(self):
#         return len(self.title)
#
#     def __getitem__(self, idx):
#         # Load audio using librosa
#         waveform, _ = librosa.load(self.audio_path[idx], sr=None)
#         title = self.title[idx]
#         return waveform, title
#
#
# # use your own data
# list_audio_path = []
# list_txt = []
# for item in input_data:
#     audio_path = audio_path + item['audio_path'].split('/')[-1]
#     caption = item['product_title'][:40]
#     list_audio_path.append(audio_path)
#     list_txt.append(caption)
#
# dataset = audio_title_dataset(list_audio_path, list_txt)
# train_dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)  # Define your own dataloader
#
# # Prepare the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6,
#                              weight_decay=0.2)  # the lr is smaller, more safe for fine tuning to new dataset
#
# # Specify the loss function
# loss_audio = nn.CrossEntropyLoss()
# loss_txt = nn.CrossEntropyLoss()
#
# # Train the model
# num_epochs = 30
# for epoch in range(num_epochs):
#     pbar = tqdm(train_dataloader, total=len(train_dataloader))
#     for batch in pbar:
#         optimizer.zero_grad()
#
#         audios, texts = batch
#
#         audios = audios.to(device)
#         texts = texts.to(device)
#
#         # Forward pass
#         logits_per_audio, logits_per_text = model(audios, texts)
#
#         # Compute loss
#         ground_truth = torch.arange(len(audios), dtype=torch.long, device=device)
#         total_loss = (loss_audio(logits_per_audio, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
#
#         # Backward pass
#         total_loss.backward()
#         optimizer.step()
#
#         pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
#
#
# # Save the model
# torch.save(model.state_dict(), 'fine_tuned_clap_model.pth') ## NEED TO DEFINE THE NAME OF THE SAVE MODEL
# # Load the model
# model = AudioClassifier(metadata_path, audio_dir, dataset_class, model_id)
# model.load_state_dict(torch.load('fine_tuned_clap_model.pth'))
#
# ################# HOW TO EVALUATE THE MODEL #################
# # Load the fine-tuned model
# model.load_state_dict(torch.load('fine_tuned_clap_model.pth'))
#
# # Create an instance of your AudioClassifier with the fine-tuned model
# audio_classifier = AudioClassifier(metadata_path, audio_dir, dataset_class, model)
#
# # Evaluate the model
# good_predictions, bad_predictions = audio_classifier.predict_and_evaluate(batch_size=1000, len_dataset=len(dataset))
#
# # Print the good and bad predictions
# print("Good predictions:", good_predictions)
# print("Bad predictions:", bad_predictions)
# # Plot the good and bad predictions for the 3 classes with the lowest accuracy
# audio_classifier.plot_predictions_by_class(good_predictions, bad_predictions)