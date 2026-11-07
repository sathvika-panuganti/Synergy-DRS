from flask import Blueprint, jsonify, request
from services.data_service import get_options
from services.prediction_service import predict

prediction_api = Blueprint("prediction", __name__)

@prediction_api.get("/options")
def options():
    return jsonify(get_options())

@prediction_api.post("/predict")
def prediction():
    body = request.get_json(silent=True) or {}
    drug1 = body.get("drug1", body.get("drug1_smiles", ""))
    drug2 = body.get("drug2", body.get("drug2_smiles", ""))
    cell_line = body.get("cell_line", "")
    missing = [label for label, value in (("drug1", drug1), ("drug2", drug2), ("cell_line", cell_line))
               if not str(value).strip()]
    if missing:
        return jsonify({"error": "Please provide both drug names or SMILES strings and a cell line.", "missing": missing}), 400
    try:
        return jsonify(predict(str(drug1).strip(), str(drug2).strip(), str(cell_line).strip()))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503
