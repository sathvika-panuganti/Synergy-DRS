import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from api.prediction import prediction_api
from services.data_service import get_dashboard_data

ROOT = BACKEND_DIR.parent
app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="/assets")
CORS(app)
app.register_blueprint(prediction_api, url_prefix="/api")

@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/style.css")
def style():
    return send_from_directory(app.static_folder, "style.css")

@app.get("/script.js")
def script():
    return send_from_directory(app.static_folder, "script.js")

@app.get("/api/dashboard")
def dashboard():
    return jsonify(get_dashboard_data())

if __name__ == "__main__":
    app.run(debug=False, port=5000)
