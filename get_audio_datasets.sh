#!/bin/bash

# Create a directory for the dataset if it doesn't exist
[ -d audios ] || mkdir audios

# Download ESC-50
if [ "$(find audios/ESC-50/ -type f | wc -l)" -eq 0 ]; then
    echo "Downloading ESC-50..."
    [ -d audios/ESC-50 ] || mkdir audios/ESC-50
    curl -L -o audios/ESC-50/ESC-50.zip https://github.com/karoldvl/ESC-50/archive/master.zip
    unzip audios/ESC-50/ESC-50.zip -d audios/ESC-50/
    rm audios/ESC-50/ESC-50.zip
    echo "ESC-50 has been downloaded."
else
    echo "ESC-50 Dataset has been already downloaded."
fi

# Download GTZAN
# NOTE: GTZAN url is not available anymore, so it needs to be downloaded manually
if [ "$(find audios/GTZAN/ -type f | wc -l)" -eq 0 ]; then
    echo "Downloading GTZAN..."
    [ -d audios/GTZAN ] || mkdir audios/GTZAN
    curl -L -o audios/GTZAN/GTZAN.zip http://opihi.cs.uvic.ca/sound/genres.tar.gz
    unzip audios/GTZAN/GTZAN.zip -d audios/GTZAN/
    rm audios/GTZAN/GTZAN.zip
    echo "GTZAN has been downloaded."
else
    echo "GTZAN Dataset has been already downloaded."
fi

echo "Downloading Audio Datasets, this operation can take a couple of minutes..."
# Download FSD50K
if [ "$(find audios/FSD50K/ -type f | wc -l)" -eq 0 ]; then
    echo "Downloading FSD50K..."
    [ -d audios/FSD50K ] || mkdir audios/FSD50K
    curl -L -C - -o audios/FSD50K/FSD50K.dev_audio.zip https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip
    unzip audios/FSD50K/FSD50K.dev_audio.zip -d audios/FSD50K/
    rm audios/FSD50K/FSD50K.dev_audio.zip
    echo "FSD50K has been downloaded."
else
    echo "FSD50K Dataset has been already downloaded."
fi

chmod -R 777 audios