"""
Kokoro-82M Neural TTS Engine for NeuraMind Deep Learning Workshop
Produces human-like, high-fidelity neural speech (24kHz).
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

import wave
import torch
import soundfile as sf
from kokoro import KPipeline


class KokoroTTS:
    def __init__(self, voice="am_michael", lang_code="a"):
        self.voice = voice
        self.lang_code = lang_code
        self.sample_rate = 24000
        print(f"Initializing Kokoro-82M Neural Engine (voice='{self.voice}')...")
        self.pipeline = KPipeline(lang_code=self.lang_code, repo_id="hexgrad/Kokoro-82M")

    def generate_audio(self, text: str, output_path: str) -> float:
        """
        Synthesizes speech using Kokoro-82M and returns duration in seconds.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        generator = self.pipeline(text, voice=self.voice)
        
        # Collect all audio chunks for this text segment
        audio_chunks = []
        for _, _, audio in generator:
            chunk = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
            audio_chunks.append(chunk)

        if not audio_chunks:
            raise RuntimeError(f"Kokoro failed to generate audio for text: {text}")

        # Concatenate chunks if text was split into multiple sentences
        import numpy as np
        full_audio = np.concatenate(audio_chunks)

        # Write 24kHz WAV
        sf.write(output_path, full_audio, self.sample_rate)

        # Measure exact duration
        with wave.open(output_path, "r") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)

        return duration


if __name__ == "__main__":
    tts = KokoroTTS(voice="am_michael")
    test_out = "testing_samples/kokoro_output/test_michael.wav"
    dur = tts.generate_audio("Welcome to the NeuraMind Deep Learning workshop.", test_out)
    print(f"Generated test file: {test_out} ({dur:.2f}s)")

