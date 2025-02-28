import streamlit as st
import pyaudio
import wave
import io
import numpy as np

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
RATE = 16000
FRAMES_PER_BUFFER = 3200

# Gain slider (1.0 = normal, >1.0 = amplify)
gain = st.sidebar.slider("Gain (Volume Boost)", min_value=1.0, max_value=10.0, step=0.1, value=2.0)

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

col1, col2 = st.columns(2)
col1.button("Start Recording", on_click=start_listening)
col2.button("Stop Recording", on_click=stop_listening)

# Record audio
if st.session_state.run:
    st.write(f"🎤 Recording from: **{selected_device}** (Index: {device_index})")

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER, input_device_index=device_index)

    for _ in range(100):  # Record short chunks
        audio_data = stream.read(FRAMES_PER_BUFFER)
        st.session_state.audio_frames.append(audio_data)

    stream.stop_stream()
    stream.close()

# Convert stored frames into a WAV file and apply gain
if not st.session_state.run and st.session_state.audio_frames:
    raw_audio = b"".join(st.session_state.audio_frames)
    amplified_audio = apply_gain(raw_audio, gain=gain)  # Apply gain

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(amplified_audio)

    wav_buffer.seek(0)
    st.audio(wav_buffer, format="audio/wav")

st.write("Press 'Start Recording' to capture audio and 'Stop Recording' to play it back.")
