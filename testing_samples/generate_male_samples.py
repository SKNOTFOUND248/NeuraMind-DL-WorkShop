import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import soundfile as sf
import wave
from kokoro import KPipeline

print("=" * 60)
print(" Kokoro-82M Voice Sample Generator: Male Voices")
print("=" * 60)

pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

output_dir = os.path.join(os.path.dirname(__file__), "kokoro_output")
os.makedirs(output_dir, exist_ok=True)

test_voices = [
    {
        "id": "am_adam",
        "name": "Adam (American Male - Clear & Engaging)",
        "text": "Hello, I am Adam. Welcome to the NeuraMind Deep Learning Workshop. Today we explore the mathematical foundations of the perceptron.",
        "filename": "male_sample_adam.wav"
    },
    {
        "id": "am_michael",
        "name": "Michael (American Male - Deep & Authoritative)",
        "text": "Hello, I am Michael. In deep learning, understanding the linear decision boundary is the first step toward mastering complex neural architectures.",
        "filename": "male_sample_michael.wav"
    },
    {
        "id": "bm_george",
        "name": "George (British Male - Articulate & Formal)",
        "text": "Hello, I am George. Every modern neural network began with Frank Rosenblatt's perceptron in nineteen fifty-seven.",
        "filename": "male_sample_george.wav"
    }
]

generated_samples = []

for voice_info in test_voices:
    voice_id = voice_info["id"]
    print(f"\n[Generating] {voice_info['name']} (voice='{voice_id}')...")
    
    generator = pipeline(voice_info["text"], voice=voice_id)
    
    for i, (gs, ps, audio) in enumerate(generator):
        out_path = os.path.join(output_dir, voice_info["filename"])
        audio_data = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
        sf.write(out_path, audio_data, 24000)
        
        with wave.open(out_path, 'r') as wf:
            dur = wf.getnframes() / float(wf.getframerate())
            size_kb = os.path.getsize(out_path) / 1024
            
        print(f"  ✓ Saved to: {out_path}")
        print(f"  ✓ Duration: {dur:.2f}s | Size: {size_kb:.1f} KB | Rate: 24kHz")
        generated_samples.append({
            "voice": voice_id,
            "label": voice_info["name"],
            "path": out_path,
            "duration": round(dur, 2)
        })

print("\n" + "=" * 60)
print("All male voice samples generated successfully!")
print("=" * 60)

