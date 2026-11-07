"""
NeuraMind Deep Learning Masterclass: In-Depth Audio Narration Generator
Uses Kokoro-82M with the deep authoritative male voice 'am_michael' at speed=0.88
for deliberate, university-grade explanations of Perceptron architecture,
matrix dot product flow, student placement math, and NumPy code.
"""

import os
import json
import soundfile as sf
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from kokoro import KPipeline

AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "audio"))
os.makedirs(AUDIO_DIR, exist_ok=True)

MASTER_SCRIPT_V2 = [
    {
        "act": 1,
        "title": "The Concrete Problem: Student Placement & Feature Space",
        "voice_script": (
            "Welcome to NeuraMind. To understand the Perceptron with genuine mathematical depth, "
            "we begin not with abstractions, but with a real classification challenge: predicting college student placement. "
            "Consider a dataset with two continuous input features: a student's I Q, denoted as x 1, and their college C G P A, "
            "denoted as x 2. The target label y is binary: one represents Placed, while zero represents Not Placed. "
            "When we plot these students into a two dimensional coordinate space with I Q on the horizontal axis and C G P A on the vertical axis, "
            "the data naturally forms two clusters. The fundamental objective of the perceptron is to automatically discover a dividing boundary "
            "that cleanly isolates placed students from unplaced students."
        ),
        "subtitle": "Task: Predict Placement (y in {0,1}) from IQ (x1) and CGPA (x2) in 2D Feature Space"
    },
    {
        "act": 2,
        "title": "Artificial Neuron Architecture & Matrix Dot Product Flow",
        "voice_script": (
            "How does the artificial neuron process this information? Let us examine the linear algebra and matrix data flow. "
            "Each student is represented as an input feature vector x with dimension D. We introduce a weight vector w of matching dimension, "
            "and a scalar bias b. In compact matrix notation, we augment the input vector by prepending a constant one at index zero. "
            "This combines the bias directly into our augmented weight vector. The forward pass computes the vector dot product: "
            "z equals w transpose times x plus b, which is the sum of each feature multiplied by its corresponding weight. "
            "Consider a test applicant with I Q of 100 and C G P A of 5.1. With candidate weights of 1.0 for I Q, 2.0 for C G P A, "
            "and a bias of minus 105.0, the dot product z evaluates to positive 5.2. Passing this value through the Heaviside step activation function, "
            "since 5.2 is greater than zero, the output evaluates to one, predicting placement. Notice also feature importance: "
            "because the weight for C G P A is double that of I Q, changes in C G P A have twice the mathematical impact on the final decision."
        ),
        "subtitle": "Augmented Matrix Flow: z = w~^T x~ = b + w1*x1 + w2*x2 -> Step(z) -> y_hat"
    },
    {
        "act": 3,
        "title": "Geometric Decision Boundary & Slope-Intercept Form",
        "voice_script": (
            "Now let us examine the geometry of this calculation. The decision boundary occurs exactly where the dot product z equals zero. "
            "In our two dimensional feature space, this equation is w 1 x 1 plus w 2 x 2 plus b equals zero. "
            "By rearranging this algebraic expression into standard slope intercept form, x 2 equals minus w 1 over w 2 times x 1, "
            "minus b over w 2. This reveals the precise physical meaning of our model parameters. The slope m is governed entirely "
            "by the ratio of the weights, determining the orientation and angle of the decision boundary. The vertical intercept c "
            "is governed by the bias b, which shifts the line parallel across the coordinate plane without altering its angle. "
            "Furthermore, the weight vector w acts as the normal vector perpendicular to the line, pointing directly into the positive half space "
            "where predictions evaluate to one, while the opposing half space represents class zero."
        ),
        "subtitle": "Boundary: w1*x1 + w2*x2 + b = 0 => x2 = -(w1/w2)*x1 - (b/w2) | Normal w points to z >= 0"
    },
    {
        "act": 4,
        "title": "Learning Mechanics, Error Discrepancy & Vector Updates",
        "voice_script": (
            "How does an untrained perceptron discover optimal weights? Through the Perceptron Learning Rule. "
            "Suppose an untrained line misclassifies a placed student with label one as unplaced with prediction zero. "
            "The error term, y minus y hat, evaluates to plus one. To correct this, we update parameters by adding the learning rate eta "
            "multiplied by the input coordinates. This vector addition rotates and pulls the decision boundary directly toward the misclassified sample. "
            "Conversely, if an unplaced student is misclassified into the positive region, the error term evaluates to minus one, "
            "subtracting the vector and pushing the boundary away. When predictions are correct, the error is zero and parameters remain unchanged. "
            "Mathematically, this unified update rule corresponds to executing stochastic gradient descent on the Perceptron Loss function, "
            "which penalizes samples proportional to their distance across the wrong side of the margin."
        ),
        "subtitle": "Unified Rule: w_new = w_old + eta*(y - y_hat)*x | SGD on Loss L = max(0, -y*f(x))"
    },
    {
        "act": 5,
        "title": "Real-Time Training Simulation & Parameter Evolution",
        "voice_script": (
            "Let us observe this training process in action across real student data. We initialize the perceptron with arbitrary random weights. "
            "Initially, the boundary cuts through the dataset, producing multiple misclassifications highlighted by pulsing red indicators. "
            "In the first iteration, an error triggers an update, adjusting the weight vector and shifting the decision line. "
            "In the second iteration, another misclassified point pulls the boundary further into alignment, reducing errors to one. "
            "Finally, the remaining error is resolved. The boundary settles cleanly between the two classes, achieving zero misclassifications "
            "and complete linear convergence."
        ),
        "subtitle": "Simulation: Live parameter updates resolve errors 4 -> 2 -> 1 -> 0 (Convergence)"
    },
    {
        "act": 6,
        "title": "Line-by-Line Python NumPy Implementation",
        "voice_script": (
            "Now let us examine how this theory translates into production Python code using NumPy. "
            "In the init method, we define the learning rate and training epochs. "
            "In the fit method, we first call n p dot insert to prepend a column of ones to our feature matrix, augmenting it for the bias term. "
            "We initialize our parameter vector w to zeros. In the training loop, for each training instance, we compute the scalar dot product z, "
            "apply the Heaviside step threshold to produce y hat, and execute the unified weight update rule whenever an error occurs. "
            "Finally, the predict method processes entire batches of unseen test students simultaneously using vectorized matrix multiplication."
        ),
        "subtitle": "Code: np.insert(bias) -> dot product z -> Step threshold -> Unified update -> Batch predict"
    },
    {
        "act": 7,
        "title": "Convergence Graph, Generalization & The XOR Failure",
        "voice_script": (
            "Examining the training curve on the right, we observe the classification error plunge rapidly to zero as the model converges. "
            "On the left, the shaded decision regions demonstrate model generalization, proving that any new student applicant is accurately "
            "mapped to their appropriate class. However, we must confront the fundamental limitation of a single perceptron: "
            "it can only solve linearly separable problems. When presented with non linear distributions such as the X O R problem, "
            "where positive instances lie diagonally opposite, a single straight line can never succeed. "
            "To overcome this barrier, deep learning stacks multiple artificial neurons into hidden layers, forming Multi-Layer Perceptrons "
            "capable of bending decision surfaces to approximate any complex mathematical function."
        ),
        "subtitle": "Convergence to 0 Loss | Generalization Mesh | Stacking Perceptrons Solves Non-Linear XOR"
    }
]

def generate_all_narration_v2(voice: str = "am_michael", speed: float = 0.88) -> str:
    print("=" * 65)
    print(f"Generating In-Depth Kokoro Narration ({voice}) at speed={speed} for 7 Acts")
    print("=" * 65)
    
    pipeline = KPipeline(lang_code="a")
    manifest = []
    audio_segments = []
    
    for item in MASTER_SCRIPT_V2:
        act_num = item["act"]
        wav_filename = f"act_{act_num}.wav"
        wav_path = os.path.join(AUDIO_DIR, wav_filename)
        
        print(f"Synthesizing Act {act_num}: {item['title']}...")
        generator = pipeline(item["voice_script"], voice=voice, speed=speed)
        chunks = []
        for _, _, audio in generator:
            chunks.append(audio)
            
        full_audio = np.concatenate(chunks, axis=0)
        sf.write(wav_path, full_audio, 24000)
        
        info = sf.info(wav_path)
        duration = round(info.duration, 2)
        print(f"  -> Saved {wav_filename}: {duration}s")
        
        manifest.append({
            "act": act_num,
            "title": item["title"],
            "script": item["voice_script"],
            "subtitle": item["subtitle"],
            "file": wav_filename,
            "path": wav_path,
            "duration": duration,
            "voice": voice,
            "speed": speed
        })
        audio_segments.append(full_audio)
        # Add 0.8s pause between acts for calm pacing
        audio_segments.append(np.zeros(int(24000 * 0.8), dtype=np.float32))
        
    # Write manifest.json
    manifest_path = os.path.join(AUDIO_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")
    
    # Concatenate master audio track
    master_audio = np.concatenate(audio_segments, axis=0)
    master_wav_path = os.path.join(AUDIO_DIR, "master_narration.wav")
    sf.write(master_wav_path, master_audio, 24000)
    master_info = sf.info(master_wav_path)
    print(f"Master narration audio created: {master_wav_path} ({round(master_info.duration, 2)}s)")
    return master_wav_path

if __name__ == "__main__":
    generate_all_narration_v2()
