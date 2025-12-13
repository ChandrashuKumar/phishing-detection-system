# Phishing Detection System - Backend

Flask-based REST API for phishing detection using ML models.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

### 4. Run Development Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## Project Structure

```
backend/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes/              # API endpoints
│   ├── models/              # ML model loaders
│   ├── services/            # Business logic
│   └── utils/               # Helper functions
├── app.py                   # Entry point
├── config.py                # Configuration
├── requirements.txt         # Dependencies
└── .env                     # Environment variables
```

## Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy"
}
```

## Production Deployment

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
