import os
# Configure offline cache and disable unnecessary TensorFlow initialization
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import soundfile as sf
from kokoro import KPipeline
from IPython.display import display, Audio

print("Initializing Kokoro pipeline (lang_code='a')...")
pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

text = '''
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''

print("Generating audio with voice='af_heart'...")
generator = pipeline(text, voice='af_heart')

output_dir = os.path.join(os.path.dirname(__file__), "kokoro_output")
os.makedirs(output_dir, exist_ok=True)

for i, (gs, ps, audio) in enumerate(generator):
    print(f"\n[Segment {i}]")
    print("Graphemes:", gs)
    print("Phonemes:", ps)
    
    # Save audio chunk (convert to numpy if it is a torch tensor)
    out_file = os.path.join(output_dir, f"kokoro_sample_{i}.wav")
    audio_data = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
    sf.write(out_file, audio_data, 24000)
    print(f"-> Audio saved successfully to: {out_file}")

print("\nAll audio segments generated successfully!")
