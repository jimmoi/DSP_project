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

if "expanded" not in st.session_state:
    st.session_state.expanded = False

if st.button("Expand" if not st.session_state.expanded else "Collapse"):
    st.session_state.expanded = not st.session_state.expanded  

if "fig" not in st.session_state:
    st.session_state.fig, (st.session_state.ax1, st.session_state.ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

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
RATE = 22050 #44100
FRAMES_PER_BUFFER = int(RATE*0.5) #0.25

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
            p_time_start = time.time()
            output = model(mel_spec)
            p_time_end = time.time()
            p_time = p_time_end - p_time_start
        

            probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()
        
        # Sort probabilities
        sorted_indices = np.argsort(probabilities)
        sorted_probs = probabilities[sorted_indices]
        class_labels = list(map_index_class.values())
        sorted_labels = [class_labels[i] for i in sorted_indices]

        # 🔹 เลือกข้อมูลที่จะแสดง
        top_n = -5
        if st.session_state.expanded:
            labels, probs = sorted_labels, sorted_probs  # แสดงทั้งหมด
        else:
            labels, probs = sorted_labels[top_n:], sorted_probs[top_n:]  # แสดง Top 5

        if len(labels) == 0 or len(probs) == 0:
            st.warning("No data available for plotting!")
        else:
            fig, (ax1, ax2) = st.session_state.fig, (st.session_state.ax1, st.session_state.ax2)
            ax1.clear()
            ax2.clear()
            background_color = (12/255, 14/255, 19/255)
            time_axis = np.linspace(0, 0.5, FRAMES_PER_BUFFER)
            ax1.set_facecolor(background_color)
            ax1.plot(time_axis, audio_np, color='cyan', linewidth=1.5)
            ax1.set_title("Real-Time Audio Signal", color="white")
            ax1.set_xlabel("Time (s)", color="white")
            ax1.set_ylabel("Amplitude", color="white")
            ax1.tick_params(axis='x', colors='white')
            ax1.tick_params(axis='y', colors='white')
            # Plot real-time classification probabilities
            # fig, ax = plt.subplots(figsize=(6, max(len(labels) * 0.8, 4)))
            fig.patch.set_facecolor(background_color)
            ax2.set_facecolor(background_color)

            y_positions = [i * 2 for i in range(len(labels))]

            ax2.barh(y_positions, probs, height=1, color="blue")

            ax2.set_yticks(y_positions)
            ax2.set_yticklabels(labels, color='white')
            # ax.tick_params(axis='y', colors='white')

            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Probability", color='white')
            ax2.set_title("Real-Time Classification Probabilities", color='white')

            ax2.tick_params(axis='x', colors='white')
            ax2.tick_params(axis='y', colors='white')
            fig.tight_layout()
            # Update the plot in Streamlit
            plot_placeholder.pyplot(fig)

            time.sleep(0.1)  # Update interval

    stream.stop_stream()
    stream.close()

# Show final classification result when recording stops
if not st.session_state.run and st.session_state.audio_frames:
    st.write("🎵 **Final Classification Result:**")
    st.write(f"🔹 **Predicted Instrument:** `{sorted_labels[-1]}` (Probability: {sorted_probs[-1]:.2f})")
