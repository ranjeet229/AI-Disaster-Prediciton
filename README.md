# DisastraSense

**AI-based disaster prediction and response system** — predict floods, earthquakes, and hurricanes; get safety guidance from a chatbot; and detect persons in post-disaster images.

---

## Overview

DisastraSense uses machine learning to forecast natural disasters and support response efforts. It provides:

- **Flood risk prediction** from weather, river, and terrain data  
- **Earthquake magnitude estimation** by location  
- **Hurricane maximum wind prediction** from storm and wind data  
- **Disaster & safety chatbot** (Chatbase API or built-in demo)  
- **YOLOv5 victim detection** in uploaded images  

The app runs as a **Flask web application** with a single frontend (home page → Get started → prediction tools).

---

## Features

| Feature | Description |
|--------|-------------|
| **Flood prediction** | Input latitude, longitude, rainfall, temperature, humidity, river discharge, water level, elevation, land cover, soil type, population density, infrastructure, and historical floods → **Flood risk: Yes/No** |
| **Earthquake prediction** | Input latitude and longitude → **Predicted magnitude** |
| **Hurricane prediction** | Input location, wind quadrants (NE, SE, SW, NW), and date → **Predicted maximum wind (knots)** |
| **Chatbot** | Ask about floods, earthquakes, hurricanes, or safety tips. Uses Chatbase API when configured; otherwise built-in demo replies. |
| **YOLO victim detection** | Upload an image → YOLOv5 detects and counts persons (e.g. victims); returns count and annotated image. |

---

## Tech stack

- **Backend:** Python, Flask  
- **ML:** scikit-learn, pandas, joblib (Flood, Earthquake, Hurricane models)  
- **Detection:** PyTorch, YOLOv5 (via PyTorch Hub)  
- **Chat:** Chatbase API (optional)  
- **Frontend:** HTML, CSS, JavaScript (no separate Node app)  

---

## Quick start

### 1. Clone and enter the project

```bash
git clone https://github.com/ranjeet229/AI-Disaster-Prediciton.git
cd AI-Based-Disaster-Prediction-and-Response-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create prediction models (first time only)

```bash
python create_all_models.py
```

This generates the Flood, Earthquake, and Hurricane models used by the app.

### 4. Run the app

```bash
python app.py
```

### 5. Open in browser

Go to **http://127.0.0.1:5000**

- **Home:** Smart Disaster Prediction → click **GET STARTED**  
- **App:** Use the navbar dropdown to switch between Flood, Earthquake, Hurricane, Chatbot, and YOLO Victim Detection. Toggle **Light/Dark** theme as needed.

Stop the server with `Ctrl+C`.

---

## Chatbase (optional)

To use the **real Chatbase API** instead of demo replies:

1. Sign up at [chatbase.co](https://www.chatbase.co) and create a chatbot.  
2. Get your **API key** (Workspace Settings → API Keys) and **Chatbot ID** (chatbot Settings → General).  
3. Set them before starting the app:
   - **Option A — environment variables**
     - Windows (PowerShell):  
       `$env:CHATBASE_API_KEY="your-key"; $env:CHATBASE_CHATBOT_ID="your-id"`
     - Linux/macOS:  
       `export CHATBASE_API_KEY=your-key` and `export CHATBASE_CHATBOT_ID=your-id`
   - **Option B — `.env` file** in the project root:
     ```
     CHATBASE_API_KEY=your-key
     CHATBASE_CHATBOT_ID=your-id
     ```
     (Requires `python-dotenv`; install via `pip install -r requirements.txt`.)

Restart the app after setting variables. Without them, the chatbot uses built-in demo replies.

---

## YOLO victim detection

The YOLO panel uses **YOLOv5** (loaded via PyTorch Hub). The first time you click **Analyze image**, the app downloads the `yolov5s` model (~15 MB).  
For a **CPU-only** install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Project structure

| Item | Purpose |
|------|--------|
| `app.py` | Flask app: routes, prediction APIs, chat, YOLO |
| `create_all_models.py` | Train and save Flood, Earthquake, Hurricane models |
| `templates/home.html` | Home page (DisastraSense landing) |
| `templates/main.html` | App UI (predictions, chatbot, YOLO) |
| `*.pkl` | Saved models (created by `create_all_models.py`) |
| `requirements.txt` | Python dependencies |

---

## License

See repository for license information.
