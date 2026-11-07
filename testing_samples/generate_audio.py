"""
Generates narration audio clips for the workshop video using Kokoro-82M (voice='am_michael').
"""

import os
import json
from kokoro_engine import KokoroTTS

def generate_narration(voice="am_michael"):
    tts = KokoroTTS(voice=voice)
    audio_dir = os.path.join(os.path.dirname(__file__), "audio")
    os.makedirs(audio_dir, exist_ok=True)

    scripts = [
        {
            "act": 1,
            "title": "Introduction to the Artificial Neuron",
            "text": "Welcome to NeuraMind. In deep learning, every neural network begins with an atomic computational unit: the artificial neuron.",
            "file": "act_1.wav"
        },
        {
            "act": 2,
            "title": "Forward Pass & Linear Combination",
            "text": "The neuron receives input features, multiplies them by connection weights, adds a bias, and passes the sum through an activation function.",
            "file": "act_2.wav"
        },
        {
            "act": 3,
            "title": "Geometric Decision Boundary",
            "text": "In feature space, this linear formula forms a decision boundary line, separating positive and negative classes.",
            "file": "act_3.wav"
        },
        {
            "act": 4,
            "title": "Real Learning in Action",
            "text": "When an untrained model misclassifies a point, the learning rule triggers. It adjusts weights to pull the boundary directly toward the error.",
            "file": "act_4.wav"
        },
        {
            "act": 5,
            "title": "Convergence & Deep Learning Bridge",
            "text": "Through iterative updates, the boundary converges to zero error. Stacking these simple units creates deep learning.",
            "file": "act_5.wav"
        }
    ]

    manifest = []
    print(f"\nGenerating Kokoro Neural TTS Audio ({voice})...\n")
    for item in scripts:
        out_path = os.path.join(audio_dir, item["file"])
        dur = tts.generate_audio(item["text"], out_path)
        item["duration"] = round(dur, 2)
        item["path"] = out_path
        item["voice"] = voice
        manifest.append(item)
        print(f"  [Act {item['act']}] {item['duration']}s -> {item['file']}")
        print(f"      Text: \"{item['text'][:65]}...\"")

    manifest_path = os.path.join(audio_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to: {manifest_path}")
    return manifest

if __name__ == "__main__":
    generate_narration(voice="am_michael")
