# Deep-Learning-CLAP
### This repository provides some experiments run over audio files using CLAP (Contrastive Language-Audio Pretraining)
## About this project
This project is intended to prove state-of-art of CLAP and different use cases and applications.
## Quick Start
To get started, audio files and its respective labels need to be provided. So for this, ```get_audio_datasets.sh``` must be run. This file will create an ```audios/``` directory where all audios will be stored locally.
This project has set up 3 Datasets, however, is not limited to it. New datasets can be added and extend to the ```AudioDatasets Classes```

Once Audio Data is stored and available you can run ```main.py``` file which will prompt an easy menu to interact with on the terminal to pick which experiment, dataset and CLAP model you want to run. So give it a try!

## Datasets
There are 3 datasets available:

- **ESC-550:** is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification. It comprises 2000 5s-clips of 50 different classes across natural, human and domestic sounds.
- **GTZAN:** contains 1000 tracks of 30 second length. There are 10 genres, each containing 100 tracks which are all 22050Hz Mono 16-bit audio files in .wav format. The genres are:
    blues
    classical
    country
    disco
    hiphop
    jazz
    metal
    pop
    reggae
    rock
- **MusicSentiments:** This is a custome dataset created by us. From *YouTube* we have pick one or 2 hours audios for a specifi sentiment, so we randomly search happy music and took the audio and split it in 10 seconds audios. We did this for 4 audios, getting 1626 audio files. The sentiments classes we chose were:
    happy
    sad
    epic
    suspense

## Experiments
There are 3 main experiments available on this project:

- Zero-Shot Audio Classification
- Text to audio retrieval
- Fine-Tune Audio Classification

The result of some test cases for each experiment can be found in ```experiments/runnin_experimentst.ipynb``` this *Jupyter Notebook* gives a great understanding about the results and performing of each experiments for each task. Inside the *Notebok* you can find the disscusion about each experiments and diferent escenarios with their respective results, so from there you can replicate a similar implementation by your own.