# DisastraSense

**AI-based disaster prediction system** — predict floods, earthquakes, and hurricanes from your inputs.

---

## Overview

DisastraSense uses machine learning to forecast natural disasters. It provides:

- **Flood risk prediction** from weather, river, and terrain data  
- **Earthquake magnitude estimation** by location  
- **Hurricane maximum wind prediction** from storm and wind data  

The app runs as a **Flask web application** (home page → Get started → prediction tools).

---

## Features

| Feature | Description |
|--------|-------------|
| **Flood prediction** | Input latitude, longitude, rainfall, temperature, humidity, river discharge, water level, elevation, land cover, soil type, population density, infrastructure, and historical floods → **Flood risk: Yes/No** |
| **Earthquake prediction** | Input latitude and longitude → **Predicted magnitude** |
| **Hurricane prediction** | Input location, wind quadrants (NE, SE, SW, NW), and date → **Predicted maximum wind (knots)** |

---

## Tech stack

- **Backend:** Python, Flask  
- **ML:** scikit-learn, pandas, joblib (Flood, Earthquake, Hurricane models)  
- **Frontend:** HTML, CSS, JavaScript  

---

## Quick start

### 1. Clone and enter the project

```bash
git clone https://github.com/ranjeet229/AI-Disaster-Prediciton.git
cd AI-Disaster-Prediciton
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
- **App:** Use the navbar dropdown to switch between Flood, Earthquake, and Hurricane prediction. Toggle **Light/Dark** theme as needed.

Stop the server with `Ctrl+C`.

---

## Project structure

| Item | Purpose |
|------|--------|
| `app.py` | Flask app and prediction routes |
| `create_all_models.py` | Train and save Flood, Earthquake, Hurricane models |
| `templates/home.html` | Home page (DisastraSense landing) |
| `templates/main.html` | App UI (prediction forms) |
| `*.pkl` | Saved models (created by `create_all_models.py`) |
| `requirements.txt` | Python dependencies |

---

## License

See repository for license information.
