# 1. Setup imports
import tensorflow as tf
import librosa
import numpy as np
import gradio as gr
import json

# 2. Load the model and labels
model = tf.keras.models.load_model('speech_command_cnn.keras')
with open('/content/labels.json', 'r') as f:
    classes = json.load(f)

# 3. Define the prediction pipeline (Self-contained)
def quick_predict(file_path):
    # Feature extraction logic
    audio, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feature = tf.image.resize(mel_db[..., np.newaxis], [64, 64])
    feature = (feature - tf.reduce_min(feature)) / (tf.reduce_max(feature) - tf.reduce_min(feature))

    # Predict
    prediction = model.predict(tf.expand_dims(feature, axis=0), verbose=0)[0]
    return {classes[i]: float(prediction[i]) for i in range(len(classes))}

# 4. Launch UI
demo = gr.Interface(
    fn=quick_predict,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Label(num_top_classes=5),
    title="Pre-trained Voice AI"
)
demo.launch(share=True)
