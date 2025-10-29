# 🛫 ATCOSIM Air Traffic Control Simulation Speech Corpus

## 📘 Overview
The **ATCOSIM Corpus** is a speech dataset designed for **Air Traffic Control (ATC)** speech recognition research.  
It contains **simulated ATC operator utterances** recorded during real-time ATC simulations.  
All utterances are **in English**, spoken by **10 non-native controllers** using headset microphones.

- **Total Duration**: ~10 hours  
- **Audio Format**: 32 kHz, 16-bit PCM, mono WAV  
- **Language**: English (non-native)  
- **Speakers**: 10 (German/Swiss; male and female)  
- **Utterances**: ~10,078  

---

## 👩‍✈️ Speaker Profiles

| ID  | Nationality | Native Tongue | Gender | Sector    | # Utterances |
|-----|--------------|----------------|---------|------------|---------------|
| sm1 | German       | German          | Male    | Söllingen  | 1167 |
| sm2 | German       | German          | Male    | Söllingen  | 1848 |
| sm3 | German       | German          | Male    | Söllingen  | 808 |
| sm4 | German       | German          | Male    | Söllingen  | 1162 |
| gf1 | Swiss        | French          | Female  | Geneva     | 238 |
| gm1 | Swiss        | French          | Male    | Geneva     | 384 |
| gm2 | Swiss        | French          | Male    | Geneva     | 378 |
| zf1 | Swiss        | German          | Female  | Zürich     | 1716 |
| zf2 | Swiss        | German          | Female  | Zürich     | 1739 |
| zf3 | Swiss        | German          | Female  | Zürich     | 638 |

---

## 📂 Directory Structure

ATCOSIM/
├── DOC/ # Documentation (PDF, license, report)
├── WAVdata/ # All speech audio files
│ ├── sm1/ # Speaker ID
│ │ ├── sm1_01/ # Session 1
│ │ │ ├── sm1_01_001.wav
│ │ │ ├── sm1_01_002.wav
│ │ │ └── ...
│ │ ├── sm1_02/
│ │ └── ...
│ └── zf3/... # Up to zf3_XX_YYY.wav
│
├── TXTdata/ # Orthographic transcriptions (plain text)
│ ├── sm1/sm1_01/sm1_01_001.txt
│ └── ...
│ └── fulldata.csv # Master table of metadata + transcriptions
│
├── HTMLdata/ # Readable HTML transcription tables
│ ├── fulldata_static.htm
│ ├── fulldata_dynamic.htm
│ └── ...
│
└── wordlist.txt # Complete word vocabulary (ATC codes, callsigns, etc.)

---

## 🧾 fulldata.csv Description

Each row in `fulldata.csv` corresponds to **one utterance**, containing:
- File ID (e.g., `zf2_04_010`)
- Speaker ID / Session / Utterance #
- Orthographic transcription
- Additional metadata (time, duration, control sector, etc.)

---

## 🎧 Data Properties

- **Recording Environment**: Real-time ATC simulations  
- **Channel**: Close-talk headset  
- **Noise Level**: Low (but some simulation background)  
- **Accent Variation**: Moderate (non-native English)  
- **Speech Type**: Command-oriented, concise, domain-specific (e.g. "Speedbird two four climb flight level two six zero")

---

## ⚙️ Usage Recommendations

### 🧠 Speech Recognition Tasks
- End-to-End ASR training (Whisper / Wav2Vec2 / ESPnet / Kaldi)
- Domain adaptation for ATC communication
- Acoustic and language modeling in constrained vocabulary environments

### 📊 Preprocessing
1. Resample audio to 16 kHz, mono.
2. Normalize volume.
3. Align `.wav` with `.txt` by filename.
4. Load metadata from `fulldata.csv`.

### 📜 Example JSON for ASR training
```json
{
  "audio": "WAVdata/sm1/sm1_01/sm1_01_001.wav",
  "text": "Speedbird two four climb flight level two six zero",
  "speaker": "sm1",
  "session": "sm1_01"
}
