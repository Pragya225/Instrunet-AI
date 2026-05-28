# 🎵 InstruNet AI - Music Instrument Recognition System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([YOUR_APP_URL_HERE](https://instrunet-ai-5kzjmzgibp4qgfsphwjhre.streamlit.app/))
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

> **Advanced Multi-Task Deep Learning System for Automated Audio Analysis**

InstruNet AI is an intelligent music instrument recognition platform powered by a custom Multi-Task Convolutional Neural Network. The system simultaneously predicts **instrument type**, **audio quality**, and **recording condition** from uploaded audio files with **96% accuracy**.

---
## 🚀 Live Demo

**Try it now:** https://instrunet-ai-5kzjmzgibp4qgfsphwjhre.streamlit.app/

---

## 🌟 Key Features

### 🎯 **Triple-Task AI Analysis**
- **🎸 Instrument Classification**: 8 acoustic instruments (brass, flute, guitar, keyboard, mallet, reed, string, vocal) - **96.12% accuracy**
- **🎚️ Quality Assessment**: 4-level grading (excellent → good → fair → poor) - **99.95% accuracy**
- **🕰️ Condition Analysis**: Era detection (modern/vintage) + Signal quality (clean/noisy) - **100% accuracy**

### 🔐 **Secure User System**
- User registration & login with email validation
- Bcrypt password hashing (industry-standard security)
- Persistent cookie-based sessions
- Protected routes & authentication middleware

### 🎨 **UI/UX**
- **Dual Theme**: Seamless dark/light mode toggle
- **Animated Gauges**: Radial confidence meters with smooth transitions
- **Live Visualizations**: Real-time waveform & mel-spectrogram rendering
- **Responsive Design**: Perfect on desktop, tablet, and mobile

### 📊 **Comprehensive Analytics**
- Top 3 predictions with confidence breakdowns
- Detailed quality scoring across all 4 levels
- Era & noise analysis with sub-metrics
- Audio metadata (duration, sample rate, total samples)

### 📥 **Professional Exports**
- **JSON Reports**: Machine-readable format for data pipelines
- **PDF Reports**: Presentation-ready documents with branding

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      InstruNet AI Pipeline                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  User Upload (WAV/MP3/OGG)                                     │
│         ↓                                                      │
│  Authentication Layer (bcrypt + YAML)                          │
│         ↓                                                      │
│  Audio Processing (librosa)                                    │
│    ├─ Load at 22050 Hz                                         │
│    ├─ Standardize to 3 seconds                                 │
│    └─ Convert to Mel-Spectrogram (128×128)                     │
│         ↓                                                      │
│  Multi-Task CNN (TensorFlow)                                   │
│    ├─ Shared Backbone (4 Conv blocks)                          │
│    │   ├─ Conv2D(32→64→128→256)                                │
│    │   ├─ BatchNorm + MaxPool                                  │
│    │   └─ Dropout (0.3-0.5)                                    │
│    │                                                           │
│    └─ Task-Specific Heads                                      │
│        ├─ Head 1: Instrument (8 classes)                       │
│        ├─ Head 2: Quality (4 classes)                          │
│        └─ Head 3: Condition (4 classes)                        │
│         ↓                                                      │
│  Results Rendering                                             │
│    ├─ Radial gauges + confidence bars                          │
│    ├─ Waveform & spectrogram plots                             │
│    └─ Export to JSON/PDF                                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```
**Instrument Classification Report:**
```
Instrument     Precision  Recall  F1-Score  Support
──────────────────────────────────────────────────
Brass              99.4%   97.4%    98.4%      800
Flute              91.5%   93.1%    92.3%      800
Guitar             97.6%   96.6%    97.1%      800
Keyboard           96.8%   97.4%    97.1%      800
Mallet             96.0%   96.3%    96.1%      800
Reed               89.3%   91.5%    90.4%      800
String             98.9%   96.8%    97.8%      800
Vocal             100.0%  100.0%   100.0%      800
──────────────────────────────────────────────────
Overall            96.2%   96.1%    96.1%     6400
```
---

## 🎯 Usage Guide

 **Register / Login**  
   Secure authentication with bcrypt-hashed passwords.

 **Upload Audio**  
   Supports WAV, MP3, OGG, FLAC, and M4A formats.

 **Analyze**  
   Multi-task CNN processes the audio (~2–5 seconds inference).

 **View Results**
   - Instrument prediction (Top 3 with confidence scores)
   - Quality assessment (Excellent → Good → Fair → Poor)
   - Condition analysis (Modern/Vintage, Clean/Noisy)

 **Export Reports**
   - **JSON** – Machine-readable format for APIs & pipelines  
   - **PDF** – Presentation-ready professional report

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit + Custom CSS | Interactive UI, responsive design |
| **Authentication** | streamlit-authenticator | User management, sessions |
| **Security** | bcrypt | Password hashing |
| **ML Framework** | TensorFlow 2.13, Keras | Model training & inference |
| **Audio Processing** | librosa 0.10 | Audio loading, mel-spectrograms |
| **Visualization** | Matplotlib | Waveform & spectrogram plots |
| **Model Storage** | Hugging Face Hub | Free CDN for model files |
| **Report Generation** | ReportLab | PDF export |
| **Data Format** | JSON, YAML | Config & export |
| **Deployment** | Streamlit Cloud | Free hosting with CI/CD |

---

## 👨‍💻 Author

**Pragya**  

- 🔗 GitHub: [Pragya225](https://github.com/Pragya2257)  
- 🔗 LinkedIn: [Pragya Tyagi](https://www.linkedin.com/in/pragya-tyagi-b22581338) 
---
