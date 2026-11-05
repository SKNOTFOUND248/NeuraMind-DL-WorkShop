"""
Master Orchestrator: Generate Native TTS Audio, Render Manim Scene, and Mux Narrated Video
"""

import os
import sys
import wave
import subprocess
import json
from generate_audio import generate_narration


def merge_wavs(wav_files, output_path):
    """Concatenate multiple WAV files sequentially using standard library wave module."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with wave.open(output_path, "wb") as outfile:
        for idx, f in enumerate(wav_files):
            with wave.open(f, "rb") as infile:
                if idx == 0:
                    outfile.setparams(infile.getparams())
                outfile.writeframes(infile.readframes(infile.getnframes()))

    with wave.open(output_path, "r") as wf:
        total_sec = wf.getnframes() / float(wf.getframerate())
    return total_sec


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, "audio")
    exports_dir = os.path.join(base_dir, "exports")
    os.makedirs(exports_dir, exist_ok=True)

    print("=" * 60)
    print(" NeuraMind Workshop: Building Narrated Sample Visualization")
    print("=" * 60)

    # 1. Generate or verify TTS Audio
    manifest_path = os.path.join(audio_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("\n[Step 1/3] Generating Native TTS audio narrations...")
        manifest = generate_narration()
    else:
        print("\n[Step 1/3] Found existing TTS audio manifest.")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    # Merge into a single master audio track
    act_wavs = [os.path.join(audio_dir, f"act_{i}.wav") for i in range(1, 6)]
    master_audio = os.path.join(audio_dir, "master_narration.wav")
    total_audio_sec = merge_wavs(act_wavs, master_audio)
    print(f"-> Master narration audio created: {master_audio} ({total_audio_sec:.2f}s)")

    # 2. Render Manim Scene
    print("\n[Step 2/3] Rendering Manim Scene (DeepLearningAtomScene)...")
    scene_script = os.path.join(base_dir, "dl_learning_scene.py")
    scene_name = "DeepLearningAtomScene"

    # We use -qm (720p 30fps) for crisp, professional quality with fast render time
    cmd_render = [
        sys.executable,
        "-m", "manim",
        "-qm",
        "--media_dir", os.path.join(base_dir, "media"),
        scene_script,
        scene_name
    ]
    print("Running command:", " ".join(cmd_render))
    proc = subprocess.run(cmd_render, capture_output=True, text=True)
    if proc.returncode != 0:
        print("Manim rendering failed!")
        print(proc.stderr)
        sys.exit(1)

    print("Manim render completed successfully!")

    # Find the rendered video
    rendered_video_dir = os.path.join(base_dir, "media", "videos", "dl_learning_scene", "720p30")
    rendered_mp4 = os.path.join(rendered_video_dir, f"{scene_name}.mp4")

    if not os.path.exists(rendered_mp4):
        print(f"Error: Could not locate rendered video at {rendered_mp4}")
        sys.exit(1)

    print(f"-> Raw video rendered at: {rendered_mp4}")

    # 3. Mux Video + Audio with FFmpeg
    print("\n[Step 3/3] Muxing video and native TTS audio with FFmpeg...")
    final_output = os.path.join(exports_dir, "deep_learning_atom_narrated.mp4")

    cmd_mux = [
        "ffmpeg",
        "-y",
        "-i", rendered_mp4,
        "-i", master_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        final_output
    ]
    mux_proc = subprocess.run(cmd_mux, capture_output=True, text=True)
    if mux_proc.returncode != 0:
        print("FFmpeg muxing failed!")
        print(mux_proc.stderr)
        sys.exit(1)

    file_size_mb = os.path.getsize(final_output) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"SUCCESS! Narrated video created successfully:")
    print(f"  File: {final_output}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Duration: ~{total_audio_sec:.1f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()

