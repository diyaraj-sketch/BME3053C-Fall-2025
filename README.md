# BME3053C - Computer Applications for Biomedical Engineering

## 📊 Interactive Fourier Transform Explorer

An educational Streamlit application for exploring Fourier transforms interactively. Designed for students in **BME3053C** to understand signal decomposition, frequency analysis, sampling theory, and spectral processing.

---

## 🎯 Overview

This app transforms complex Fourier transform concepts into interactive, hands-on learning. Students can:

- **Build custom signals** from sinusoid components or presets
- **Visualize time-frequency relationships** with linked plots
- **Explore aliasing effects** and the Nyquist theorem
- **Learn windowing techniques** and spectral leakage reduction
- **Apply frequency-domain filtering** via band reconstruction

All with **embedded educational explanations** at every step.

---

## 🚀 Quick Start

### Installation

1. **Navigate to the workspace:**
   ```bash
   cd /workspaces/BME3053C-Fall-2025
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open in browser:**
   The app will open automatically, or visit `http://localhost:8501`

---

## 📚 Feature Sets

### 📈 Tab 1: Signal Builder

**Purpose:** Compose complex signals from simple sinusoid components.

**Features:**
- **5 Sinusoid Slots**: Each with independent controls (amplitude, frequency, phase, waveform, enable/disable)
- **Preset Waveforms**: Square Wave, Triangle Wave, Sawtooth, Gaussian Pulse, Step, Impulse, Chirp
- **Signal Expression**: LaTeX display showing the mathematical formula
- **Time-Domain Plot**: Interactive visualization with hover details

**Educational Value:**
- Understand superposition principle: complex signals = sum of sinusoids
- Visualize how amplitude, frequency, and phase combine
- Recognize why sharp transitions require high-frequency components

---

### 🔄 Tab 2: Time-Frequency Linked Views

**Purpose:** Bridge time-domain and frequency-domain representations.

**Features:**
- **Side-by-Side Plots**: Time-domain (left) and frequency magnitude/phase (right)
- **Interactive Hover**: Cross-reference between time and frequency domains
- **Band Select & Reconstruct**: Filter frequency ranges and overlay reconstructed signal

**Educational Value:**
- Grasp Fourier duality: time ↔ frequency representations
- Understand filtering: frequency-domain operations have time-domain effects
- See why sharp events require many frequencies

---

### ⏱️ Tab 3: Sampling, Aliasing & Resolution

**Purpose:** Master the relationship between sampling rate and signal fidelity.

**Features:**
- **Sampling Controls**: Rate, demo frequency, Nyquist calculation
- **Sampled vs. Continuous Plot**: Visual comparison of discrete vs. ideal signals
- **Aliasing Demonstration**: Automatic detection and warning when fs is too low
- **Zero-Padding Controls**: Understand resolution vs. information tradeoff

**Educational Value:**
- **Nyquist-Shannon Theorem**: Sample at ≥ 2× highest frequency
- **Aliasing**: How undersampling creates phantom frequencies
- **Real-world**: Medical device sampling rates (ECG: 250+ Hz, EEG: 250+ Hz)

---

### 🪟 Tab 4: Windowing & Spectral Leakage

**Purpose:** Reduce spectral artifacts and improve frequency localization.

**Features:**
- **Window Selection**: Rectangular, Hann, Hamming, Blackman, Kaiser
- **Kaiser β Tuning**: Adjust tradeoff between main-lobe and sidelobe
- **Spectral Leakage Demo**: Misalign frequency to see energy spreading
- **Spectral Metrics**: Main-lobe width and peak sidelobe level

**Educational Value:**
- **Spectral Leakage**: Frequency misalignment causes energy spreading
- **Window Tradeoffs**: Main-lobe width vs. sidelobe level
- **Biomedical**: Detecting weak signals (arrhythmias, seizures) amid noise

---

## 🎓 Biomedical Applications

### ECG Signal Processing
- **Sampling**: 250–1000 Hz typical
- **Filtering**: Band-pass 0.5–40 Hz removes baseline and EMI
- **Windowing**: Hann window reduces leakage in spectral analysis

### EEG Signal Analysis
- **Frequency Bands**: Delta (0.5–4 Hz), Theta, Alpha, Beta, Gamma
- **Windowing**: Critical for artifact rejection and seizure detection

---

## 📊 Course Materials for BME3053C

