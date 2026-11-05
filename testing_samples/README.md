# Testing Samples: Manim Animation with Native TTS Audio

This directory contains a complete, automated end-to-end pipeline demonstrating **3Blue1Brown-style mathematical animations using Manim Community Edition synchronized with offline Native Windows TTS audio**.

---

## 🎬 Generated Output

- **Final Narrated Video:** [`deep_learning_atom_narrated.mp4`](file:///d:/Neuramind/DL_workshop/testing_samples/exports/deep_learning_atom_narrated.mp4)
- **Resolution:** 720p 30fps (`-qm` for fast, crisp workshop display)
- **Duration:** ~43.3 seconds
- **Audio:** Native Windows SAPI Speech Synthesis (offline, high clarity, AAC 192kbps)

---

## 🧠 Storyline & Visual Acts

| Act | Topic | Key Animation Elements | Audio Narration |
|:---:|:---|:---|:---|
| **Act 1** | **The Atom of Learning** | Title card, NeuraMind branding, smooth zoom-in | *"Welcome to NeuraMind. In deep learning, every neural network begins with an atomic computational unit: the artificial neuron."* |
| **Act 2** | **Forward Pass & Soma** | Inputs ($x_1, x_2$), synaptic weights ($w_1, w_2$), bias ($b$), signal pulses into $\Sigma$, step activation $f(z)$ | *"The neuron receives input features, multiplies them by connection weights, adds a bias, and passes the sum through an activation function."* |
| **Act 3** | **Geometric Decision Boundary** | 2D feature space axes ($x_1$ vs $x_2$), scatter data points, decision line $w_1 x_1 + w_2 x_2 + b = 0$, half-space partition | *"In feature space, this linear formula forms a decision boundary line, separating positive and negative classes."* |
| **Act 4** | **Real Learning in Action** | Untrained line misclassifying a point, alert pulse, perceptron update rule formula ($w \leftarrow w + \eta (y - \hat{y}) x$), boundary rotation and translation | *"When an untrained model misclassifies a point, the learning rule triggers. It adjusts weights to pull the boundary directly toward the error."* |
| **Act 5** | **Convergence & Deep Learning** | Zero-error badge, glowing boundary, transition from 1 neuron to multi-layer deep network | *"Through iterative updates, the boundary converges to zero error. Stacking these simple units creates deep learning."* |

---

## 📁 Directory Structure

```
testing_samples/
├── README.md                      # Documentation & instructions (This file)
├── tts_engine.py                  # Zero-dependency Native Windows SAPI TTS synthesizer
├── generate_audio.py              # Generates act-by-act WAVs and manifest.json
├── dl_learning_scene.py           # Manim Community Edition scene implementation
├── build_narrated_video.py        # Master build pipeline (Audio + Manim + FFmpeg)
├── audio/                         # Generated speech files
│   ├── act_1.wav to act_5.wav     # Individual act audio clips
│   ├── master_narration.wav       # Merged synchronized audio track
│   └── manifest.json              # Exact duration & transcript metadata
├── exports/
│   └── deep_learning_atom_narrated.mp4 # Final synchronized narrated video
└── media/                         # Manim intermediate render cache
```

---

## 🚀 How to Re-run or Customize

1. **Re-generate audio & render video in one command:**
   ```bash
   python testing_samples/build_narrated_video.py
   ```

2. **Render in 1080p60 High Quality:**
   Edit the flag in `build_narrated_video.py` from `-qm` to `-qh`, then run:
   ```bash
   python testing_samples/build_narrated_video.py
   ```

