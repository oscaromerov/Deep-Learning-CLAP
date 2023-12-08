import json
import os
import pandas as pd

def load_json_config(file_name: str):
    file_path = f"configurations/{file_name}"
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return None

# def add_directory_name_to_files(folder_path: str):
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         if os.path.isfile(file_path) and filename.endswith(".wav"):
#             directory_name = os.path.basename(folder_path)
#             new_file_name = f"{directory_name}_{filename}"
#             os.rename(file_path, os.path.join(folder_path, new_file_name))
#     print("File renaming complete.")
#
# def add_directory_name_to_csv(csv_path: str):
#     df = pd.read_csv(csv_path)
#     directory_name = os.path.basename(os.path.dirname(csv_path))
#     # df['Audio_File'] = [f"{directory_name}_{filename}" for filename in df['Audio_File'].values]
#     df['Sentiment'] = [filename.lower() for filename in df['Sentiment'].values]
#     df.to_csv(csv_path, index=False)
#     print("CSV file updated.")
#
# def merge_csv_files(input_dirs, output_file):
#     merged_data = pd.DataFrame()
#     for input_dir in input_dirs:
#         for file_name in os.listdir(input_dir):
#             if file_name.endswith(".csv"):
#                 file_path = os.path.join(input_dir, file_name)
#                 data = pd.read_csv(file_path)
#                 merged_data = pd.concat([merged_data, data], ignore_index=True)
#     merged_data.to_csv(output_file, index=False)
#     print(f"Merged data saved to {output_file}")

# # Specify the input directories containing the CSV files
# input_dirs = ["audios/emotions/epic", "audios/emotions/happy", "audios/emotions/sad", "audios/emotions/suspense"]
#
# # Specify the path for the output merged CSV file
# output_file = "audios/emotions/emotions_music_labels.csv"
#
# # Call the function to merge the CSV files
# merge_csv_files(input_dirs, output_file)
# add_directory_name_to_csv("audios/emotions/emotions_music_labels.csv")
#
# # Specify the path to the folder containing the audio files
# folder_path = "audios/emotions/suspense"
#
# # Call the function to add the directory name to the files
# add_directory_name_to_files(folder_path)
# add_directory_name_to_csv(f'{folder_path}/suspense_music_labels.csv')
