from flask import Blueprint, request, jsonify, send_from_directory, current_app
from .yolo_classifier import process_image, get_sorted_predictions
from dotenv import load_dotenv
import os
import json
import tempfile

# Charger les variables d'environnement depuis .env (localement)
load_dotenv()

# Créer un Blueprint
routes_bp = Blueprint('routes', __name__)

# Chemins de base avec fallback sur les variables d'environnement
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.path.join(BASE_DIR, '../uploads'))
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', os.path.join(BASE_DIR, '../static/results'))
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(BASE_DIR, '../models'))
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', os.path.join(BASE_DIR, '../weights/best.pt'))

# Gestion de serviceAccountKey.json
SERVICE_ACCOUNT_KEY = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
if SERVICE_ACCOUNT_KEY:
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(SERVICE_ACCOUNT_KEY)
        GOOGLE_APPLICATION_CREDENTIALS = temp_file.name
else:
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', os.path.join(BASE_DIR, 'serviceAccountKey.json'))

# Créer les dossiers si inexistants
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@routes_bp.route('/', methods=['POST'])
def welcome():
    return jsonify({'message': 'Bienvenue à l\'API Flask pour la prédiction des plantes !'}), 200

@routes_bp.route('/prediction_des_plantes/', methods=['POST'])
def predict_plants():
    print("Contenu de request.files :", request.files)
    if 'file' not in request.files:
        print("Aucun fichier détecté dans request.files")
        return jsonify({'error': 'Aucun fichier uploadé'}), 400

    file = request.files['file']
    print("Nom du fichier :", file.filename)
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    if file:
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
        file.save(upload_path)
        print(f"Image sauvegardée à : {upload_path}")

        try:
            output_path = process_image(upload_path, current_app.config['OUTPUT_FOLDER'], MODEL_DIR, WEIGHTS_PATH)
            return jsonify({'result': f'/results/{os.path.basename(output_path)}'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@routes_bp.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename)

@routes_bp.route('/predictions', methods=['GET'])
@routes_bp.route('/predictions/', methods=['GET'])
def get_predictions():
    sorted_predictions = get_sorted_predictions()
    return jsonify(sorted_predictions)