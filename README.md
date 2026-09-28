# Speech Command AI

This project demonstrates how to build and deploy a simple Convolutional Neural Network (CNN) for recognizing spoken commands using the Google Speech Commands Dataset v0.02. The entire workflow, from data download and preprocessing to model training and deployment with Gradio, is covered within this Colab notebook.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Workflow](#workflow)
  - [1. Library Installation and Imports](#1-library-installation-and-imports)
  - [2. Dataset Download and Preparation](#2-dataset-download-and-preparation)
  - [3. Feature Extraction and TensorFlow Dataset Creation](#3-feature-extraction-and-tensorflow-dataset-creation)
  - [4. CNN Model Architecture and Training](#4-cnn-model-architecture-and-training)
  - [5. Model Evaluation and Prediction](#5-model-evaluation-and-prediction)
  - [6. Gradio UI Deployment](#6-gradio-ui-deployment)
- [Usage](#usage)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Credits and License](#credits-and-license)

## Project Overview

The goal of this project is to classify single spoken words into a predefined set of command categories (e.g., "yes", "no", "up", "down"). It leverages `librosa` for audio processing, `TensorFlow/Keras` for model building, and `Gradio` for creating an interactive web UI to test real-time audio predictions.

## Dataset

The project uses the **Google Speech Commands Dataset v0.02**, which consists of one-second `.wav` audio files of spoken English words. The dataset includes 35 different words, with a focus on core commands like "yes", "no", "go", "stop", etc., as well as auxiliary words and background noise samples.

More details about the dataset can be found in the original `README.md` file located at `/content/README.md` after extraction.

## Setup

To run this notebook, you don't need any special API keys. Just ensure you have a Google Colab environment or a Python environment with the following libraries installed:

```bash
pip install -q librosa soundfile gradio tensorflow
```

This is handled automatically in the first cell of the notebook.

## Workflow

The notebook follows these main steps:

### 1. Library Installation and Imports

Essential libraries like `librosa`, `soundfile`, `gradio`, and `tensorflow` are installed and imported at the beginning of the notebook.

### 2. Dataset Download and Preparation

- The Google Speech Commands Dataset v0.02 is downloaded and extracted to the Colab environment.
- The `classes` list (representing the unique spoken words) is extracted from the dataset's directory structure.
- A Pandas DataFrame is created to map file paths to their corresponding labels.

### 3. Feature Extraction and TensorFlow Dataset Creation

- A custom `extract_features` function is defined to convert raw audio (`.wav`) into Mel Spectrograms, which are then resized and normalized to be used as input for the CNN.
- `tf.data.Dataset` is used to create an efficient data pipeline for training and validation, performing on-the-fly feature extraction to conserve memory.

### 4. CNN Model Architecture and Training

- A Convolutional Neural Network (CNN) is built using Keras, designed to classify the Mel Spectrograms.
- The model is compiled with an `Adam` optimizer and `sparse_categorical_crossentropy` loss.
- The model is trained using the prepared `tf.data.Dataset` pipelines, with `EarlyStopping` to prevent overfitting.

### 5. Model Evaluation and Prediction

- The trained model's performance is evaluated on the test set.
- Functions are provided to predict on individual audio files, including recording custom audio directly from the browser.

### 6. Gradio UI Deployment

- A user-friendly web interface is created using `Gradio`, allowing for real-time speech command predictions either by speaking into the microphone or uploading an audio file.

## Usage

1.  **Run all cells:** Execute the entire notebook in sequence.
2.  **Interact with Gradio UI:** Once the Gradio UI cell (Part 12 or 13, depending on the notebook version) is executed, a local URL and a public URL (if `share=True`) will be provided. Open the public URL in your browser.
3.  **Speak or Upload:** Use the microphone to speak a command or upload a `.wav` file.
4.  **Get Predictions:** Click the "Predict Word" button to see the model's prediction and confidence.

## Saving and Loading the Model

- The trained Keras model is saved to `speech_command_cnn.keras`.
- The `classes` (labels) are saved to `labels.json`.
- Instructions are provided within the notebook on how to download these files and load them in a separate environment or notebook for inference.

## Credits and License

- **Dataset:** Google Speech Commands Dataset v0.02
  - Citation: P. Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition", arXiv e-prints, 2018. ([arXiv:1804.03209](https://arxiv.org/abs/1804.03209))
  - License: Creative Commons BY 4.0 license
- **Frameworks:** TensorFlow, Keras, Librosa, Gradio
