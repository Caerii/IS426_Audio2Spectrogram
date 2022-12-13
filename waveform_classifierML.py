# This is a project that takes in waveform time series (audio, EEG, EMG, etc.)
# and batch processes and normalizes the data
# waveform -> MEL Spectrogram -> CNN recognition -> classification

# First we have to create a folder structure to store the data
# We're going to create a folder for each class of data
# Let's clone the existing data into a new folder
import os
import numpy as np
import shutil

# Relative file path
inputDataPath = './data/ingestion/'

# if the file path does not exist, write it
if not os.path.exists(inputDataPath):
    os.makedirs(inputDataPath)

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()


# If the folder is empty, ask the user to put in a file to copy from
if not os.listdir(inputDataPath):
    print("Please put a file in the data/ingestion folder")
    file_path = filedialog.askopenfilename()
    # use the file_path selected and copy over the files inside into ingestion (they should all be audio files)
    shutil.copy(file_path, inputDataPath)

# First we're going to need to create a function that takes in a waveform and normalizes it into chunkSize second chunks, according to the sampling rate
import soundfile as sf

def waveformIngestion(filePath, chunkSize, sampleRate):
    # Read in the file
    data, fs = sf.read(filePath)
    # Check the sampling rate
    if fs != sampleRate:
        print("Sampling rate does not match")
        pass
    # Check the length of the data
    if len(data) < chunkSize * sampleRate:
        print("Data is too short")
        pass
    # Check the number of channels
    if len(data.shape) > 1:
        print("Data is not mono")
        pass
    # if data is not mono, force it to be
    if len(data.shape) > 1:
        data = data[:,0]
    # Now we're going to split the data into chunks
    # First we're going to create a list of the start and end points of each chunk
    chunkList = []
    for i in range(0, len(data), chunkSize * sampleRate):
        chunkList.append([i, i + chunkSize * sampleRate])
    # Now we're going to create a list of the chunks
    chunkData = []
    for i in chunkList:
        chunkData.append(data[i[0]:i[1]])
    # Now we're going to return the chunk data
    return chunkData

# Let's use our function on an audio file to test it
# First we're going to create a folder for the copied audio files
cloneDataPath = './data/processing/'
if not os.path.exists(cloneDataPath):
    os.makedirs(cloneDataPath)
# Now we're going to copy the files into the folder
import shutil
for file in os.listdir(inputDataPath):
    shutil.copy(inputDataPath + file, cloneDataPath + file)
# Now we're going to create a folder for the normalized data
normalizedDataPath = './data/normalized/'
if not os.path.exists(normalizedDataPath):
    os.makedirs(normalizedDataPath)
# Now we're going to normalize the data
for file in os.listdir(cloneDataPath):
    # First we're going to create a folder for the file
    if not os.path.exists(normalizedDataPath + file):
        os.makedirs(normalizedDataPath + file)
    # Now we're going to normalize the data
    chunkData = waveformIngestion(cloneDataPath + file, 1, 44100)
    # Now we're going to write the data to the folder
    for i in range(len(chunkData)):
        sf.write(normalizedDataPath + file + '/' + str(i) + '.wav', chunkData[i], 44100)

# Now we're going to create a folder for the spectrograms
spectrogramDataPath = './data/spectrogram/'
if not os.path.exists(spectrogramDataPath):
    os.makedirs(spectrogramDataPath)
# Now we're going to create the spectrograms
import librosa
import librosa.display
import matplotlib.pyplot as plt
for file in os.listdir(normalizedDataPath):
    # First we're going to create a folder for the file
    if not os.path.exists(spectrogramDataPath + file):
        os.makedirs(spectrogramDataPath + file)
    # Now we're going to create the spectrograms
    for i in os.listdir(normalizedDataPath + file):
        # Read in the file
        data, fs = sf.read(normalizedDataPath + file + '/' + i)
        # Create the spectrogram
        S = librosa.feature.melspectrogram(y=data, sr=fs, n_mels=128, fmax=8000)
        # Convert to log scale
        log_S = librosa.power_to_db(S, ref=np.max)
        # Save the spectrogram
        plt.imsave(spectrogramDataPath + file + '/' + i[:-4] + '.png', log_S)

# Now we're going to create a folder for the training data
trainingDataPath = './data/training/'
if not os.path.exists(trainingDataPath):
    os.makedirs(trainingDataPath)
# Now we're going to create the training data
for file in os.listdir(spectrogramDataPath):
    # First we're going to create a folder for the file
    if not os.path.exists(trainingDataPath):
        os.makedirs(trainingDataPath)
    # Now we're going to create the training data
    for i in os.listdir(spectrogramDataPath + file):
        # Read in the file
        data = plt.imread(spectrogramDataPath + file + '/' + i)
        # Save the training data
        np.save(trainingDataPath + file + '_' + i[:-4], data)

# Now we're going to create a folder for the testing data
testingDataPath = './data/testing/'
if not os.path.exists(testingDataPath):
    os.makedirs(testingDataPath)
# Now we're going to create the testing data
for file in os.listdir(spectrogramDataPath):
    # First we're going to create a folder for the file
    if not os.path.exists(testingDataPath):
        os.makedirs(testingDataPath)
    # Now we're going to create the testing data
    for i in os.listdir(spectrogramDataPath + file):
        # Read in the file
        data = plt.imread(spectrogramDataPath + file + '/' + i)
        # Save the testing data
        np.save(testingDataPath + file + '_' + i[:-4], data)

# Now we're going to create a folder for the validation data
validationDataPath = './data/validation/'
if not os.path.exists(validationDataPath):
    os.makedirs(validationDataPath)
# Now we're going to create the validation data
for file in os.listdir(spectrogramDataPath):
    # First we're going to create a folder for the file
    if not os.path.exists(validationDataPath):
        os.makedirs(validationDataPath)
    # Now we're going to create the validation data
    for i in os.listdir(spectrogramDataPath + file):
        # Read in the file
        data = plt.imread(spectrogramDataPath + file + '/' + i)
        # Save the validation data
        np.save(validationDataPath + file + '_' + i[:-4], data)

