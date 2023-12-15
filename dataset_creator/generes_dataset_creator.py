import os
import pandas as pd
from pydub import AudioSegment

os.chdir('/Users/oscar/repositories/Deep-Learning-CLAP')

# Step 1: Define Input Audios and Output Folder
input_audios = {
    'blues': 'audios/Blues.mp3',
    'classical': 'audios/Classical.mp3',
    'contry': 'audios/Contry.mp3',
    'disco': 'audios/Disco.mp3',
    'hiphop' : "audios/Hiphop.mp3",
    'jazz': "audios/Jazz.mp3",
    'metal': "audios/Metal.mp3",
    'pop': "audios/Pop.mp3",
    'reggae': "audios/Reggae.mp3",
    'rock': "audios/Rock.mp3"
}

output_folder = "audios/GenresDataSet200Samp"

# Step 2: Create Output Folder if Not Exists
os.makedirs(output_folder, exist_ok=True)

# Step 3: Extract Audio Segments and Create CSV
def extract_audio_segments(audio_path, sentiment, output_folder, segment_duration=10, starting_number=1, max_segments=250):
    audio = AudioSegment.from_mp3(audio_path)
    total_duration = len(audio)
    segment_start = 0
    segment_number = starting_number
    csv_data = []

    while segment_start < total_duration and segment_number - starting_number < max_segments:
        segment_end = min(segment_start + segment_duration * 1000, total_duration)
        audio_segment = audio[segment_start:segment_end]
        segment_filename = f"audio_segment_{segment_number}.wav"
        segment_path = os.path.join(output_folder, segment_filename)

        # Record information in the CSV file
        csv_data.append({
            'filename': segment_filename,
            'label': sentiment
        })

        # Export audio segment as WAV file
        audio_segment.export(segment_path, format='wav')

        segment_start += segment_duration * 1000
        segment_number += 1

    return csv_data, segment_number

# Process each audio and accumulate CSV data
all_csv_data = []
overall_segment_count = 1

for sentiment, audio_path in input_audios.items():
    csv_data, overall_segment_count = extract_audio_segments(audio_path, sentiment, output_folder, starting_number=overall_segment_count, max_segments=250)
    all_csv_data.extend(csv_data)

# Create CSV File
csv_file_path = os.path.join(output_folder, "all_music_labels.csv")
df = pd.DataFrame(all_csv_data)
df.to_csv(csv_file_path, index=False)

print(f"Audio segments and CSV file created successfully at: {csv_file_path}")