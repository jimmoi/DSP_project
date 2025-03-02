import sys
import os
import streamlit as st
import pyaudio
import wave
import io
sys.path.append(os.getcwd())
import numpy as np
from model import test_model
import torch
import torch.nn.functional as F
import librosa
import pickle
from model.train_model.model import InstrumentClassifier_CBAM
import time
import matplotlib.pyplot as plt

# Initialize session state
if "run" not in st.session_state:
    st.session_state.run = False
if "audio_frames" not in st.session_state:
    st.session_state.audio_frames = []

# Start/Stop button logic
def start_listening():
    st.session_state.run = True
    st.session_state.audio_frames = []

def stop_listening():
    st.session_state.run = False

# Apply Gain (Increase Volume)
def apply_gain(audio_bytes, gain=2.0):
    """ Increase volume by a gain factor (default 2x) """
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)  # Convert to NumPy array
    audio_np = np.clip(audio_np * gain, -32768, 32767).astype(np.int16)  # Amplify and prevent clipping
    return audio_np.tobytes()  # Convert back to bytes

# Initialize PyAudio
p = pyaudio.PyAudio()

# List available microphones
device_list = []
device_indices = []
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info["maxInputChannels"] > 0:
        device_list.append(device_info["name"])
        device_indices.append(i)

# UI: Select a microphone
selected_device = st.sidebar.selectbox("Select a Microphone", device_list, index=0)
device_index = device_indices[device_list.index(selected_device)]

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FRAMES_PER_BUFFER = int(0.25 * RATE)

with open(r"model\train_model\encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
onehot = encoder["encoder"]
map_index_class = encoder["map_index_class"]

model = InstrumentClassifier_CBAM(28)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.load_state_dict(torch.load(r'D:\git_project\DSP_project\model\instrument_classifier_cbam_200_0.0001_weight.pt'))
model.eval()

# Gain slider (1.0 = normal, >1.0 = amplify)
gain = st.sidebar.slider("Gain (Volume Boost)", min_value=1.0, max_value=10.0, step=0.1, value=2.0)

col1, col2 = st.columns(2)
col1.button("Start Recording", on_click=start_listening)
col2.button("Stop Recording", on_click=stop_listening)

# Create a placeholder for real-time probability visualization
plot_placeholder = st.empty()

# Record audio and update probabilities in real-time
if st.session_state.run:
    st.write(f"🎤 Recording from: **{selected_device}** (Index: {device_index})")

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER, input_device_index=device_index)

    while st.session_state.run:
        audio_data = stream.read(FRAMES_PER_BUFFER)
        st.session_state.audio_frames.append(audio_data)

        # Convert stored frames into a WAV file and apply gain
        # raw_audio = b"".join(st.session_state.audio_frames[-20])  # Use last 20 frames for real-time update
        amplified_audio = apply_gain(audio_data, gain=gain)

        # Convert audio to numpy array
        audio_np = np.frombuffer(amplified_audio, dtype=np.int16).astype(np.float32) / 32768.0
        print(audio_np.shape)
        audio_np = librosa.util.normalize(audio_np)
        mel_spec = test_model.audio_to_melspectrogram(audio_np)
        mel_spec = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(mel_spec)
            probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()

        # Sort probabilities
        sorted_indices = np.argsort(probabilities)
        sorted_probs = probabilities[sorted_indices]
        class_labels = list(map_index_class.values())
        sorted_labels = [class_labels[i] for i in sorted_indices]

        # Plot real-time classification probabilities
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(sorted_labels, sorted_probs, color="blue")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Real-Time Classification Probabilities")

        # Update the plot in Streamlit
        plot_placeholder.pyplot(fig)

        time.sleep(0.1)  # Update interval

    stream.stop_stream()
    stream.close()

# Show final classification result when recording stops
if not st.session_state.run and st.session_state.audio_frames:
    st.write("🎵 **Final Classification Result:**")
    st.write(f"🔹 **Predicted Instrument:** `{sorted_labels[-1]}` (Probability: {sorted_probs[-1]:.2f})")
