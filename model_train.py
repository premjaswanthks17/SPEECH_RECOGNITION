# ============================================
# IMPORT LIBRARIES
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)
print("Librosa version:", librosa.__version__)

# ============================================
# DOWNLOAD GOOGLE SPEECH COMMANDS DATASET V0.02
# ============================================

# Remove the old mini_speech_commands directory if it exists
!rm -rf /content/mini_speech_commands

# Download Google Speech Commands Dataset v0.02
!wget -q https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

# Extract the dataset
!tar -xf speech_commands_v0.02.tar.gz

print("Google Speech Commands Dataset v0.02 downloaded and extracted.")


import os

# ============================================
# UPDATE DATASET PATH AND CLASSES
# ============================================

DATASET_PATH = "/content/"

# Filter out non-directory entries like README.md and other non-command folders
# The original dataset has _background_noise_ and LICENSE files, so we filter those out
classes = sorted([name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name)) and not name.startswith('_') and name not in ['sample_data', '__MACOSX', '.config', '.gradio']])

print("Updated Classes:")
for c in classes:
    print(c)

print(f"\nTotal classes: {len(classes)}")

# Update label_to_index mapping for the new classes
label_to_index = {label: index for index, label in enumerate(classes)}
print("\nUpdated Label to Index Mapping:")
print(label_to_index)

# ============================================
# VERIFY DATASET EXTRACTION PATH
# ============================================

# List contents of /content/ to find the correct directory name
!ls -F /content/

print("\nLook for a directory related to 'speech_commands' in the output above.")
print("The expected directory name is usually 'speech_commands' or 'speech_commands_v0.02'.")

# ============================================
# PREPARE NEW DATAFRAME FOR GOOGLE SPEECH COMMANDS DATASET
# ============================================

data = []

for label in classes:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder):
        print(f"Skipping '{label}' as it is not a directory.")
        continue

    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    print(f"{label}: {len(files)} files")

    for file in files:
        data.append({
            "file": os.path.join(folder, file),
            "label": label
        })

df = pd.DataFrame(data)

print("\nTotal audio files in new dataset:", len(df))


def preprocess_audio(file_path, label_str):
    # Extract features using the previously defined function
    features = extract_features(file_path)

    # Convert label string to index using the dictionary defined earlier
    # We decode the bytes from the tensor
    label_name = label_str.numpy().decode('utf-8')
    label_idx = label_to_index[label_name]

    return features, label_idx

# Create the dataset pipeline
dataset = tf.data.Dataset.from_tensor_slices((df['file'].values, df['label'].values))
dataset = dataset.shuffle(len(df), seed=42, reshuffle_each_iteration=False)

train_size = int(0.8 * len(df))
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Map the preprocessing function
def map_func(file_path, label):
    return tf.py_function(preprocess_audio, [file_path, label], (tf.float32, tf.int32))

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(map_func, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)
test_dataset = test_dataset.map(map_func, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)

print("Datasets initialized successfully.")

import tensorflow as tf
import librosa
import numpy as np

# ============================================
# CREATE TENSORFLOW DATASET
# ============================================

IMG_HEIGHT = 64
IMG_WIDTH = 64

def extract_features(file_path):
    # Ensure file_path is a Python string
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode('utf-8')

    audio, sr = librosa.load(
        file_path,
        sr=16000
    )

    # Create Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=64
    )

    # Convert to decibels
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Resize to fixed dimensions
    mel_db = tf.image.resize(
        mel_db[..., np.newaxis],
        [IMG_HEIGHT, IMG_WIDTH]
    )

    # Normalize features to [0, 1]
    mel_db = (mel_db - tf.reduce_min(mel_db)) / (tf.reduce_max(mel_db) - tf.reduce_min(mel_db))

    return mel_db

def preprocess_audio(file_path, label):
    # Extract features using the previously defined function
    features = extract_features(file_path)

    # Convert label to one-hot encoding if needed, or just integer
    label_idx = tf.cast(label_to_index[label.numpy().decode('utf-8')], tf.int32)

    return features, label_idx


# Create a TensorFlow dataset from the DataFrame
# Map file paths to features and labels
dataset = tf.data.Dataset.from_tensor_slices(
    (df['file'].values, df['label'].values)
)

# Shuffle the dataset before splitting
DATASET_SIZE = len(df)
SHUFFLE_BUFFER_SIZE = DATASET_SIZE # Or a smaller buffer if memory is an issue
dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE, seed=42, reshuffle_each_iteration=False)

# Determine split sizes (e.g., 80% train, 20% test)
train_size = int(0.8 * DATASET_SIZE)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Define the output signature for tf.py_function
output_signature = (
    tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32)
)

# Apply preprocessing to both train and test datasets
# Use tf.py_function for `preprocess_audio` as it involves `librosa.load`
AUTOTUNE = tf.data.AUTOTUNE

def apply_preprocess_map(file_path, label_str):
    features, label_idx = tf.py_function(
        preprocess_audio,
        [file_path, label_str],
        Tout=[spec.dtype for spec in output_signature] # Changed 'output_signature' to 'Tout'
    )
    # Explicitly ensure shapes after py_function
    features.set_shape(output_signature[0].shape)
    label_idx.set_shape(output_signature[1].shape)
    return features, label_idx

train_dataset = train_dataset.map(
    apply_preprocess_map,
    num_parallel_calls=AUTOTUNE
)

test_dataset = test_dataset.map(
    apply_preprocess_map,
    num_parallel_calls=AUTOTUNE
)

# Further batch and prefetch for performance
BATCH_SIZE = 32

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("TensorFlow datasets created and preprocessed.")

# Verify a batch shape
for features_batch, labels_batch in train_dataset.take(1):
    print(f"\nBatch of features shape: {features_batch.shape}")
    print(f"Batch of labels shape: {labels_batch.shape}")
    print(f"Example feature value range: {tf.reduce_min(features_batch):.4f} to {tf.reduce_max(features_batch):.4f}")

import os

# ============================================
# CHECK DATASET
# ============================================

# This cell now reflects the current DATASET_PATH and classes
# as defined in previous steps for the Google Speech Commands Dataset v0.02.

print(f"Current DATASET_PATH: {DATASET_PATH}")
print("Classes:")
for c in classes:
    print(c)

print(f"\nTotal classes: {len(classes)}")


# ============================================
# COUNT AUDIO FILES
# ============================================

data = []

for label in classes:

    folder = os.path.join(DATASET_PATH, label)

    # Check if the current item is a directory before proceeding
    if not os.path.isdir(folder):
        print(f"Skipping '{label}' as it is not a directory.")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".wav")
    ]

    print(f"{label}: {len(files)} files")

    for file in files:
        data.append({
            "file": os.path.join(folder, file),
            "label": label
        })

df = pd.DataFrame(data)

print("\nTotal audio files:", len(df))

# ============================================
# FIX DATASET AND LOAD ONE AUDIO FILE
# ============================================

# 1. Ensure the path is correct
DATASET_PATH = "/content/"

# 2. Re-populate the DataFrame from the correct path
data = []
for label in classes:
    folder = os.path.join(DATASET_PATH, label)
    if os.path.isdir(folder):
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        for file in files:
            data.append({"file": os.path.join(folder, file), "label": label})

df = pd.DataFrame(data)
print(f"Total audio files found: {len(df)}")

# 3. Now load the sample file (this won't crash if len > 0)
if len(df) > 0:
    sample_file = df.iloc[0]["file"]
    audio, sample_rate = librosa.load(sample_file, sr=16000)

    print("\nFile:", sample_file)
    print("Audio shape:", audio.shape)
    print("Sample rate:", sample_rate)
    print("Duration:", len(audio) / sample_rate, "seconds")
else:
    print("Error: No audio files found in /content/. Please check if the dataset was extracted.")


# ============================================
# VISUALIZE WAVEFORM
# ============================================

plt.figure(figsize=(12, 4))

librosa.display.waveshow(
    audio,
    sr=sample_rate
)

plt.title("Speech Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()


# ============================================
# CREATE MEL SPECTROGRAM
# ============================================

mel_spectrogram = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate,
    n_mels=64
)

mel_db = librosa.power_to_db(
    mel_spectrogram,
    ref=np.max
)

plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mel_db,
    sr=sample_rate,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.show()

import tensorflow as tf
import librosa
import numpy as np

# ============================================
# AUDIO → MEL SPECTROGRAM FUNCTION
# ============================================

IMG_HEIGHT = 64
IMG_WIDTH = 64

def extract_features(file_path):
    # Ensure file_path is a Python string
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode('utf-8')

    audio, sr = librosa.load(
        file_path,
        sr=16000
    )

    # Create Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=64
    )

    # Convert to decibels
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Resize to fixed dimensions
    mel_db = tf.image.resize(
        mel_db[..., np.newaxis],
        [IMG_HEIGHT, IMG_WIDTH]
    )

    # Normalize features to [0, 1]
    mel_db = (mel_db - tf.reduce_min(mel_db)) / (tf.reduce_max(mel_db) - tf.reduce_min(mel_db))

    return mel_db

features = extract_features(sample_file)

print("Feature shape:", features.shape)


# ============================================
# PREPARE DATASET
# ============================================

X = []
y = []

label_to_index = {
    label: index
    for index, label in enumerate(classes)
}

print(label_to_index)

# ============================================
# EXTRACT FEATURES FROM ALL AUDIO
# ============================================

# Fix for NameError: name 'df' is not defined, 'classes' not defined, and 'label_to_index' not defined.
# This block ensures 'DATASET_PATH', 'classes', 'label_to_index', and 'df' are available
# if previous cells defining them were not run.
# It duplicates the DataFrame creation logic from cell UTVzpAqQ_YwL and variable definitions from cell c83ef825.
import pandas as pd
import os

# Define DATASET_PATH if not already defined (from c83ef825)
if 'DATASET_PATH' not in globals():
    DATASET_PATH = "/content/"
    print(f"Warning: 'DATASET_PATH' not found. Setting it to {DATASET_PATH} within this cell.")

# Define classes if not already defined (from c83ef825)
if 'classes' not in globals() and 'classes' not in locals():
    print("Warning: 'classes' not found. Re-creating classes list within this cell.")
    classes = sorted([name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name)) and not name.startswith('_') and name not in ['sample_data', '__MACOSX', '.config', '.gradio']])
    print(f"Classes re-created in this cell: {classes}")

# Define label_to_index if not already defined (from c83ef825 or sUxTwop6ygkA)
if 'label_to_index' not in globals() and 'label_to_index' not in locals():
    print("Warning: 'label_to_index' not found. Re-creating mapping within this cell.")
    label_to_index = {label: index for index, label in enumerate(classes)}
    print(f"Label to index mapping re-created in this cell: {label_to_index}")

# Re-create df if not already defined (from UTVzpAqQ_YwL)
if 'df' not in globals() and 'df' not in locals():
    print("Warning: 'df' not found. Re-creating DataFrame from dataset within this cell.")
    local_data = []
    for label in classes:
        folder = os.path.join(DATASET_PATH, label)
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if f.endswith(".wav")]
            for file in files:
                local_data.append({"file": os.path.join(folder, file), "label": label})
    df = pd.DataFrame(local_data)
    print(f"DataFrame 'df' re-created in this cell with {len(df)} entries.")

X = []
y = []

for index, row in df.iterrows():

    try:

        feature = extract_features(row["file"])

        X.append(feature)

        y.append(
            label_to_index[row["label"]]
        )

    except Exception as e:

        print("Error:", row["file"])
        print(e)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)


import numpy as np

# ============================================
# NORMALIZE FEATURES
# ============================================

# Ensure X is a NumPy array before attempting to use .astype()
# This handles cases where X might still be a list (e.g., if previous cells weren't run correctly)
if isinstance(X, list):
    X = np.array(X)

# Check if X is an empty array before proceeding with normalization
if X.size == 0:
    print("Warning: X is an empty array. Normalization cannot be performed. Please ensure the dataset was processed correctly in prior cells.")
else:
    X = X.astype("float32")

    # Normalize features to a [0, 1] range
    # Add a small epsilon to the denominator to prevent division by zero if max equals min
    min_val = X.min()
    max_val = X.max()
    if (max_val - min_val) != 0:
        X = (X - min_val) / (max_val - min_val)
    else:
        # If all values are the same, X is already normalized (or cannot be normalized further)
        X = np.zeros_like(X) # Or handle as appropriate for your data

    print("Minimum:", X.min())
    print("Maximum:", X.max())

# ============================================
# TRAIN / TEST SPLIT
# ============================================

from sklearn.model_selection import train_test_split
import numpy as np

# Check if X and y are empty before attempting the split
if X.size == 0 or y.size == 0:
    print("Warning: X or y is empty. Cannot perform train_test_split.")
    print("Please ensure features were extracted successfully in the preceding cells.")
    X_train, X_test, y_train, y_test = np.array([]), np.array([]), np.array([]), np.array([]) # Initialize as empty arrays
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training:", X_train.shape)
    print("Testing :", X_test.shape)



# ============================================
# BUILD CNN MODEL
# ============================================

from tensorflow import keras
from tensorflow.keras import layers
import os # Import os for directory listing

# Ensure DATASET_PATH and classes are correctly defined and not empty.
# This logic ensures the model can be built even if previous cells were not run
# or if kernel state was lost or corrupted.
if 'DATASET_PATH' not in globals():
    DATASET_PATH = "/content/"
    print(f"Warning: 'DATASET_PATH' not found. Setting it to {DATASET_PATH} within this cell.")

# Check if 'classes' is undefined, empty, or not a list.
# Re-create if necessary using the same logic as in c83ef825.
if 'classes' not in globals() or not isinstance(classes, list) or len(classes) == 0:
    print("Warning: 'classes' list is not defined or is empty. Attempting to re-create it.")

    # If classes is still empty, and the archive exists, try re-extracting the dataset.
    tar_gz_path = os.path.join(DATASET_PATH, "speech_commands_v0.02.tar.gz")
    # Check for at least one expected class directory (e.g., 'yes') to decide on re-extraction
    # instead of just checking if the tar.gz exists.
    if not os.path.isdir(os.path.join(DATASET_PATH, "yes")) and os.path.exists(tar_gz_path):
        print(f"Dataset directories not found. Re-extracting from {tar_gz_path}...")
        # Use !tar -xf to extract the dataset
        os.system(f"tar -xf {tar_gz_path} -C {DATASET_PATH}") # Extract to DATASET_PATH
        print("Dataset re-extracted.")

    classes = sorted([name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name)) and not name.startswith('_') and name not in ['sample_data', '__MACOSX', '.config', '.gradio']])
    if len(classes) > 0:
        print(f"Classes re-created successfully with {len(classes)} entries: {classes}")
    else:
        # If classes is still empty, raise a more specific error to guide the user.
        raise ValueError(f"Error: 'classes' list is empty after attempting to re-create it from '{DATASET_PATH}'. Please ensure the dataset is extracted correctly.")

# Now proceed with model building, len(classes) should be > 0.
model = keras.Sequential([

    # Input
    layers.Input(
        shape=(64, 64, 1)
    ),

    # CNN Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CNN Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CNN Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert to vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Prevent overfitting
    layers.Dropout(0.3),

    # Output
    layers.Dense(
        len(classes), # This will now use the correctly populated 'classes' count
        activation="softmax"
    )
])

print("CNN model built successfully.")


# ============================================
# BUILD CNN MODEL (Updated for 35 classes)
# ============================================

# Ensure IMG_HEIGHT and IMG_WIDTH are defined
# (These were previously defined in cells ioz1SdRKAGe_ and 38cce819)
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Dynamically set num_classes based on the updated classes list
num_classes = len(classes)
print(f"Building model for {num_classes} classes...")

model = keras.Sequential([

    # Input layer
    layers.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 1)
    ),

    # CNN Block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # CNN Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # CNN Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    # Output layer updated to num_classes (35)
    layers.Dense(num_classes, activation="softmax")
])

model.summary()


# ============================================
# COMPILE MODEL
# ============================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)



import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from tensorflow import keras
from tensorflow.keras import layers

# --- Robustness block for dependencies ---
# Ensure DATASET_PATH is defined
if 'DATASET_PATH' not in globals() or not DATASET_PATH:
    DATASET_PATH = "/content/"
    print(f"Warning: 'DATASET_PATH' not defined. Setting to {DATASET_PATH}")

# Ensure classes are populated (logic similar to 1h1_xVYM2J3Q)
if 'classes' not in globals() or not isinstance(classes, list) or len(classes) == 0:
    print(f"Warning: 'classes' list is not defined or is empty. Attempting to re-create it.")
    tar_gz_path = os.path.join(DATASET_PATH, "speech_commands_v0.02.tar.gz")
    if not os.path.isdir(os.path.join(DATASET_PATH, "yes")) and os.path.exists(tar_gz_path):
        print(f"Dataset directories not found. Re-extracting from {tar_gz_path}...")
        os.system(f"tar -xf {tar_gz_path} -C {DATASET_PATH}")
        print("Dataset re-extracted.")
    classes = sorted([name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name)) and not name.startswith('_') and name not in ['sample_data', '__MACOSX', '.config', '.gradio']])
    if len(classes) > 0:
        print(f"'classes' re-created successfully with {len(classes)} entries.")
    else:
        raise ValueError(f"Error: 'classes' list is empty after attempting to re-create it from '{DATASET_PATH}'. Please ensure the dataset is extracted correctly.")

# Ensure label_to_index is populated (logic similar to c83ef825)
if 'label_to_index' not in globals() or not isinstance(label_to_index, dict) or len(label_to_index) == 0:
    print(f"Warning: 'label_to_index' not defined or empty. Re-creating mapping.")
    label_to_index = {label: index for index, label in enumerate(classes)}
    print(f"'label_to_index' re-created.")

# Ensure df is populated (logic similar to UTVzpAqQ_YwL)
if 'df' not in globals() or not isinstance(df, pd.DataFrame) or df.empty:
    print(f"Warning: 'df' not defined or is empty. Re-creating DataFrame.")
    local_data = []
    for label in classes:
        folder = os.path.join(DATASET_PATH, label)
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if f.endswith(".wav")]
            for file in files:
                local_data.append({"file": os.path.join(folder, file), "label": label})
    df = pd.DataFrame(local_data)
    print(f"DataFrame 'df' re-created with {len(df)} entries.")
    if df.empty:
        raise ValueError(f"Error: DataFrame 'df' is empty after re-creation. No audio files found.")

# Ensure IMG_HEIGHT, IMG_WIDTH are defined (logic similar to 38cce819)
if 'IMG_HEIGHT' not in globals():
    IMG_HEIGHT = 64
    print(f"Warning: 'IMG_HEIGHT' not defined. Setting to {IMG_HEIGHT}.")
if 'IMG_WIDTH' not in globals():
    IMG_WIDTH = 64
    print(f"Warning: 'IMG_WIDTH' not defined. Setting to {IMG_WIDTH}.")

# Ensure extract_features function is defined (logic similar to 38cce819)
if 'extract_features' not in globals():
    print(f"Warning: 'extract_features' function not defined. Defining it.")
    def extract_features(file_path):
        if isinstance(file_path, tf.Tensor):
            file_path = file_path.numpy().decode('utf-8')
        audio, sr = librosa.load(file_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = tf.image.resize(mel_db[..., np.newaxis], [IMG_HEIGHT, IMG_WIDTH])
        mel_db = (mel_db - tf.reduce_min(mel_db)) / (tf.reduce_max(mel_db) - tf.reduce_min(mel_db))
        return mel_db

# Ensure preprocess_audio function is defined (logic similar to 38cce819)
if 'preprocess_audio' not in globals():
    print(f"Warning: 'preprocess_audio' function not defined. Defining it.")
    def preprocess_audio(file_path, label):
        features = extract_features(file_path)
        label_idx = tf.cast(label_to_index[label.numpy().decode('utf-8')], tf.int32)
        return features, label_idx

# Ensure train_dataset and test_dataset are defined (logic similar to 38cce819)
if 'train_dataset' not in globals() or 'test_dataset' not in globals():
    print(f"Warning: 'train_dataset' or 'test_dataset' not defined. Re-creating TensorFlow datasets.")
    dataset = tf.data.Dataset.from_tensor_slices((df['file'].values, df['label'].values))
    DATASET_SIZE = len(df)
    SHUFFLE_BUFFER_SIZE = DATASET_SIZE
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE, seed=42, reshuffle_each_iteration=False)
    train_size = int(0.8 * DATASET_SIZE)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    output_signature = (
        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    AUTOTUNE = tf.data.AUTOTUNE
    def apply_preprocess_map(file_path, label_str):
        features, label_idx = tf.py_function(
            preprocess_audio,
            [file_path, label_str],
            Tout=[spec.dtype for spec in output_signature]
        )
        features.set_shape(output_signature[0].shape)
        label_idx.set_shape(output_signature[1].shape)
        return features, label_idx
    train_dataset = train_dataset.map(apply_preprocess_map, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(apply_preprocess_map, num_parallel_calls=AUTOTUNE)
    BATCH_SIZE = 32
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    print(f"TensorFlow datasets re-created and preprocessed.")
# --- End Robustness block ---

history = model.fit(
    train_dataset, # Use the new train_dataset
    validation_data=test_dataset, # Use the new test_dataset for validation
    epochs=20,
    batch_size=32,
    verbose=1
)

# ============================================
# COMPILE MODEL
# ============================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================
# TRAIN MODEL WITH EARLY STOPPING
# ============================================

from tensorflow.keras.callbacks import EarlyStopping

# Define Early Stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
)

history = model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=20,

    batch_size=32,

    verbose=1,
    callbacks=[early_stopping] # Add the early stopping callback
)


# ============================================
# TRAIN MODEL WITH EARLY STOPPING (using tf.data.Dataset)
# ============================================

from tensorflow.keras.callbacks import EarlyStopping

# Define Early Stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
)

history = model.fit(

    train_dataset, # Use the new train_dataset
    validation_data=test_dataset, # Use the new test_dataset for validation
    epochs=20,
    callbacks=[early_stopping]
)

print("Model training completed with the new dataset.")

# ============================================
# RETRY: TRAIN MODEL WITH tf.data.Dataset
# ============================================

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# We start training on the 35-class model using the data pipeline
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10, # Reduced epochs for initial run given the large dataset size
    callbacks=[early_stopping]
)

print("Model training completed.")


# ============================================
# LOSS GRAPH
# ============================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("CNN Training Loss")

plt.legend()

plt.show()


# ============================================
# EVALUATE MODEL
# ============================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: {test_loss:.4f}"
)

# ============================================
# PREDICT ONE AUDIO FILE
# ============================================

sample = X_test[0]

sample = np.expand_dims(
    sample,
    axis=0
)

prediction = model.predict(sample)

predicted_index = np.argmax(prediction)

predicted_label = classes[predicted_index]

confidence = prediction[0][predicted_index]

print("Prediction:", predicted_label)

print(
    f"Confidence: {confidence * 100:.2f}%"
)


from IPython.display import Javascript
from google.colab import output
from base64 import b64decode

def record_audio(filename="my_voice.wav", duration=2):

    js_code = """
    async function recordAudio() {

        const stream =
            await navigator.mediaDevices
            .getUserMedia({audio:true});

        const recorder =
            new MediaRecorder(stream);

        let chunks = [];

        recorder.ondataavailable =
            e => chunks.push(e.data);

        recorder.start();

        await new Promise(
            resolve =>
            setTimeout(
                resolve,
                """ + str(duration * 1000) + """
            )
        );

        recorder.stop();

        await new Promise(
            resolve =>
            recorder.onstop = resolve
        );

        stream.getTracks()
            .forEach(track => track.stop());

        const blob =
            new Blob(
                chunks,
                {type:'audio/wav'}
            );

        const reader =
            new FileReader();

        reader.readAsDataURL(blob);

        await new Promise(
            resolve =>
            reader.onloadend = resolve
        );

        return reader.result;
    }

    recordAudio();
    """

    data = output.eval_js(js_code)

    audio_data = b64decode(
        data.split(",")[1]
    )

    with open(filename, "wb") as f:
        f.write(audio_data)

    return filename


audio_file = record_audio()

print(
    "Saved:",
    audio_file
)

# ============================================
# PREDICT CUSTOM AUDIO
# ============================================

def predict_audio(file_path):

    feature = extract_features(file_path)

    # Normalize
    feature = feature.astype("float32")

    feature = (
        feature - X.min()
    ) / (
        X.max() - X.min()
    )

    feature = np.expand_dims(
        feature,
        axis=0
    )

    prediction = model.predict(
        feature,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction
    )

    predicted_word = classes[
        predicted_index
    ]

    confidence = prediction[
        0
    ][
        predicted_index
    ]

    return predicted_word, confidence

# Ensure tensorflow is imported, as 'tf.expand_dims' is used
import tensorflow as tf

# Redefine the predict_audio function locally for this cell's execution
# This will temporarily fix the AttributeError for this specific call.
# For a permanent fix, the original 'predict_audio' definition in cell 'aP1C2TSc_GcN'
# and 'get_predictions' in cell 'LF4uFIc2AGU_' should be updated.
def predict_audio(file_path):
    # Extract features. This function already returns a tf.Tensor of type float32,
    # and has normalized the spectrogram to a [0, 1] range.
    feature = extract_features(file_path)

    # The following lines caused the AttributeError and were redundant/incorrect:
    # feature = feature.astype("float32") # AttributeError: EagerTensor object has no attribute 'astype'
    # feature = (feature - X.min()) / (X.max() - X.min()) # Redundant re-normalization

    # Add a batch dimension to the feature tensor required by the model.predict method.
    feature = tf.expand_dims(
        feature,
        axis=0
    )

    # Make the prediction using the model
    prediction = model.predict(
        feature,
        verbose=0
    )

    # Get the index of the highest probability prediction
    predicted_index = np.argmax(
        prediction
    )

    # Map the predicted index back to a human-readable label
    predicted_word = classes[
        predicted_index
    ]

    # Extract the confidence for the predicted word
    confidence = prediction[
        0
    ][
        predicted_index
    ]

    return predicted_word, confidence

# Now call the (potentially redefined) predict_audio function
word, confidence = predict_audio(
    audio_file
)

print("🎤 Predicted Word:", word)

print(
    f"📊 Confidence: {confidence * 100:.2f}%"
)


def get_predictions(file_path):

    feature = extract_features(file_path)

    feature = feature.astype("float32")

    feature = (
        feature - X.min()
    ) / (
        X.max() - X.min()
    )

    feature = np.expand_dims(
        feature,
        axis=0
    )

    predictions = model.predict(
        feature,
        verbose=0
    )[0]

    result = {
        classes[i]: float(predictions[i])
        for i in range(len(classes))
    }

    return dict(
        sorted(
            result.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


# Fix: Redefining get_predictions to handle EagerTensors correctly
def get_predictions(file_path):
    # extract_features returns a tf.Tensor normalized to [0, 1]
    feature = extract_features(file_path)

    # Add batch dimension using tf.expand_dims instead of np.expand_dims
    # The .astype() and manual normalization were removed as they are redundant and caused errors
    feature = tf.expand_dims(feature, axis=0)

    predictions = model.predict(feature, verbose=0)[0]

    result = {
        classes[i]: float(predictions[i])
        for i in range(len(classes))
    }

    return dict(
        sorted(
            result.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

# Run the fixed function
get_predictions(audio_file)

# ============================================
# SAVE MODEL
# ============================================

model.save(
    "speech_command_cnn.keras"
)

print("Model saved successfully!")
