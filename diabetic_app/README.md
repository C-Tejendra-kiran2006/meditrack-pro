# MediTrack Pro 🏥
### AI-Powered Diabetes Management Platform

## Features
- ✅ **ML Risk Assessment** — Logistic Regression + Random Forest + Gradient Boosting
- ✅ **Smart AI Chatbot** — Context-aware responses using live vitals + health history
- ✅ **AI Diet Planner** — Personalised Indian diabetic meal plans
- ✅ **AI Trend Analysis** — Health trajectory insights
- ✅ **Live IoT Vitals** — Heart rate, steps, SpO₂, temperature monitoring
- ✅ **Medication Tracker** — Full CRUD with dosage/frequency management
- ✅ **Appointment Scheduler** — Doctor visit management
- ✅ **Lab Booking** — Razorpay payment integration
- ✅ **Clinical PDF Reports** — ReportLab multi-section reports with risk colouring
- ✅ **Health Analytics** — Monthly trends, risk distribution, vitals history
- ✅ **User Profiles** — Blood group, diabetes type, allergies, emergency contact

## Setup

```bash
pip install Flask Werkzeug itsdangerous python-dotenv scikit-learn pandas reportlab requests
```

Copy `.env.example` to `.env` and fill in your API keys.

## Run

```bash
# First time: train the ML model
python3 train_diabetes.py

# Start the server
python3 app.py
```

Open: http://localhost:8080

## IoT Integration (POST /api/vitals)
```json
POST /api/vitals
X-User-Id: <user_id>
{"bpm": 78, "steps": 4200, "spo2": 97.5, "temp": 36.6}
```
