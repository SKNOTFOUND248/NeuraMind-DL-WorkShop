"""
Native Windows TTS Engine using SAPI.SpVoice via win32com.
Generates offline .wav speech audio with zero external dependencies.
"""

import os
import wave
import win32com.client


class NativeTTS:
    def __init__(self, rate=0, volume=100):
        """
        rate: -10 to +10 (0 is normal speed, 1 is slightly faster)
        volume: 0 to 100
        """
        self.rate = rate
        self.volume = volume

    def generate_audio(self, text: str, output_path: str) -> float:
        """
        Synthesizes text to a .wav file and returns duration in seconds.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Rate = self.rate
        speaker.Volume = self.volume

        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        # 3 = SSFMCreateForWrite
        stream.Open(output_path, 3, False)
        speaker.AudioOutputStream = stream
        speaker.Speak(text)
        stream.Close()

        # Measure duration from WAV header
        with wave.open(output_path, "r") as wf:
            frames = wf.getnframes()
            framerate = wf.getframerate()
            duration = frames / float(framerate)

        return duration


if __name__ == "__main__":
    tts = NativeTTS(rate=0)
    test_file = "testing_samples/test_speech.wav"
    dur = tts.generate_audio("Welcome to the NeuraMind Deep Learning workshop.", test_file)
    print(f"Generated {test_file} successfully! Duration: {dur:.2f} seconds")
    if os.path.exists(test_file):
        os.remove(test_file)
