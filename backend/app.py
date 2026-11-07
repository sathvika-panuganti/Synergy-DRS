from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from api.prediction import prediction_api
from services.data_service import get_dashboard_data

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
CORS(app)
app.register_blueprint(prediction_api, url_prefix="/api")

@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/api/dashboard")
def dashboard():
    return jsonify(get_dashboard_data())

if __name__ == "__main__":
    app.run(debug=False, port=5000)
