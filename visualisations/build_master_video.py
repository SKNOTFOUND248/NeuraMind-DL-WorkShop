"""
Master Pipeline: Build Full Perceptron Masterclass Video with Kokoro Narration
Orchestrates:
  1. Audio narration check / generation (260.25s Kokoro Michael voice)
  2. Manim CE scene rendering (720p @ 30fps)
  3. FFmpeg audio/video muxing into final production MP4
"""

import os
import sys
import subprocess
import shutil

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
PYTHON_BIN = sys.executable

AUDIO_MANIFEST = os.path.join(BASE_DIR, "audio", "manifest.json")
MASTER_AUDIO = os.path.join(BASE_DIR, "audio", "master_narration.wav")
MEDIA_DIR = os.path.join(BASE_DIR, "media")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

FINAL_OUTPUT = os.path.join(EXPORTS_DIR, "perceptron_masterclass_full.mp4")

def main():
    print("=" * 65)
    print(" NeuraMind Workshop: Building Perceptron Masterclass Full Video")
    print(" Voice: Kokoro-82M (am_michael) | Quality: 720p30")
    print("=" * 65)
    
    # 1. Check or generate narration
    if not os.path.exists(MASTER_AUDIO):
        print("\n[Step 1/3] Generating Kokoro Michael audio narration...")
        gen_script = os.path.join(BASE_DIR, "generate_master_audio.py")
        subprocess.run([PYTHON_BIN, gen_script], check=True, cwd=ROOT_DIR)
    else:
        print(f"\n[Step 1/3] Master audio found: {MASTER_AUDIO}")
        
    # 2. Render Manim Scene
    print("\n[Step 2/3] Rendering Manim Scene (PerceptronMasterclassScene)...")
    scene_file = os.path.join(BASE_DIR, "perceptron_master_scene.py")
    cmd = [
        PYTHON_BIN, "-m", "manim",
        "-qm",  # 720p @ 30fps
        "--media_dir", MEDIA_DIR,
        scene_file,
        "PerceptronMasterclassScene"
    ]
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)
    
    raw_video = os.path.join(
        MEDIA_DIR, "videos", "perceptron_master_scene", "720p30", "PerceptronMasterclassScene.mp4"
    )
    if not os.path.exists(raw_video):
        raise FileNotFoundError(f"Expected raw video not found at: {raw_video}")
    print(f"-> Raw video rendered successfully: {raw_video}")
    
    # 3. Mux Video and Master Narration Audio
    print("\n[Step 3/3] Muxing video and Kokoro master audio with FFmpeg...")
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        # Check standard manim bundled ffmpeg or fallback
        ffmpeg_bin = "ffmpeg"
        
    mux_cmd = [
        ffmpeg_bin, "-y",
        "-i", raw_video,
        "-i", MASTER_AUDIO,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        FINAL_OUTPUT
    ]
    print(f"Muxing command: {' '.join(mux_cmd)}")
    subprocess.run(mux_cmd, check=True)
    
    if os.path.exists(FINAL_OUTPUT):
        size_mb = os.path.getsize(FINAL_OUTPUT) / (1024 * 1024)
        print("\n" + "=" * 65)
        print(" SUCCESS! Masterclass Full Video generated successfully:")
        print(f"   Output File: {FINAL_OUTPUT}")
        print(f"   File Size:   {size_mb:.2f} MB")
        print("   Audio Voice: am_michael (Kokoro-82M)")
        print("=" * 65)
    else:
        raise RuntimeError("Final output video was not created!")

if __name__ == "__main__":
    main()

