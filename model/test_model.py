import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import librosa
import pickle
from jupyter.model import InstrumentDataset, InstrumentClassifier_CBAM


def audio_to_melspectrogram(audio):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB
    return mel_spec_db

def predictions(sound, model, device, onehot, map_index_class):
    """need window_size = 0.5 second, and sample rate 22050
    so sample = 11025"""
    
    #transform 
    y = librosa.util.normalize(sound)
    y = audio_to_melspectrogram(y)
    y = torch.tensor(y).unsqueeze(0)
    y = y.unsqueeze(0)

    with torch.no_grad():
        y = y.to(device)
        # Get raw logits from the model
        outputs = model(y)  # Raw logits
        # Apply softmax to the logits to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        probabilities = probabilities.cpu().numpy()[0]
    
    result = {map_index_class[i]: probabilities[i]*100 for i in range(len(probabilities))}
    
    return result


if __name__ == "__main__":
    
    ## load model
    device = "cuda" if torch.cuda.is_available else "cpu"
    model = torch.load(r"H:\DSP_project\model\jupyter\instrument_classifier_cbam_200_0.0001.pt", weights_only=False)
    model.eval()

    
    with open(r"H:\DSP_project\model\jupyter\encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    onehot = encoder["onehot"]
    map_index_class = encoder["map_index_class"]
    
    ## input 
    file_path = r"H:\DSP_project\ignoredir\dataset\archive2\Train_submission\Train_submission\violin_sound (241).wav"
    sr = 22050
    t = 0.5
    y, sr = librosa.load(file_path, sr=sr)
    test_input = y[:int(t*sr)]

    ## predict
    res = predictions(sound=test_input,
                model=model,
                device=device,
                onehot=onehot,
                map_index_class=map_index_class)

    ## visualize
    x = res.keys()
    y = res.values()
    plt.figure(figsize=(20,10))
    plt.xticks(rotation=45)
    plt.bar(x,y)
    plt.show()
