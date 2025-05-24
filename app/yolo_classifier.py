import os
import logging
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import torch
import warnings
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import cloudinary.api
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialiser Firebase Admin avec le fichier de service
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Charger les variables d'environnement depuis .env
load_dotenv()

# Configurer Cloudinary avec les variables d'environnement
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

# Vérifier que les identifiants Cloudinary sont bien chargés
if not all([os.getenv('CLOUDINARY_CLOUD_NAME'), os.getenv('CLOUDINARY_API_KEY'), os.getenv('CLOUDINARY_API_SECRET')]):
    logging.error("Les identifiants Cloudinary ne sont pas définis dans le fichier .env")
    raise ValueError("Identifiants Cloudinary manquants")

# Solution définitive - Autorisation explicite de toutes les classes nécessaires
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential,
    ultralytics.nn.modules.conv.Conv,
    torch.nn.modules.conv.Conv2d,
    ultralytics.nn.modules.block.DFL,
    ultralytics.nn.modules.head.Detect,
    ultralytics.nn.modules.Bottleneck,
    ultralytics.nn.modules.C2f,
    ultralytics.nn.modules.SPPF,
    ultralytics.nn.modules.Concat,
])

# Suppression des avertissements indésirables
warnings.filterwarnings("ignore", category=UserWarning, module="keras.*")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def resize_image(image, max_size=1000):
    """Redimensionne l'image si elle dépasse la taille maximale"""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def iou(box1, box2):
    """Calcule l'Intersection over Union entre deux boîtes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0


def fuse_boxes(boxes, iou_threshold=0.6):
    """Fusionne les boîtes qui se chevauchent"""
    fused = []
    used = [False] * len(boxes)
    for i, box1 in enumerate(boxes):
        if used[i]:
            continue
        group = [box1]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if not used[j] and iou(box1, boxes[j]) > iou_threshold:
                group.append(boxes[j])
                used[j] = True
        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)
        fused.append((x1, y1, x2, y2))
    return fused


def get_label_position(img, box, line1, line2, font_scale, thickness_text, padding, existing_labels,
                       label_iou_threshold=0.1):
    """Détermine la position optimale pour l'étiquette"""
    x1, y1, x2, y2 = box
    img_h, img_w = img.shape[:2]
    (line1_width, line1_height), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)
    (line2_width, line2_height), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)
    bg_width = max(line1_width, line2_width) + 2 * padding
    bg_height = line1_height + line2_height + 3 * padding

    positions = [
        {'bg_x1': x1, 'bg_y1': y1 - bg_height, 'text_y1': y1 - bg_height + line1_height + padding,
         'text_y2': y1 - bg_height + line1_height + line2_height + 3 * padding // 2},
        {'bg_x1': x1, 'bg_y1': y2, 'text_y1': y2 + line1_height + padding,
         'text_y2': y2 + line1_height + line2_height + 3 * padding // 2},
        {'bg_x1': max(0, x1 - bg_width), 'bg_y1': y1, 'text_y1': y1 + line1_height + padding,
         'text_y2': y1 + line1_height + line2_height + 3 * padding // 2},
        {'bg_x1': x2, 'bg_y1': y1, 'text_y1': y1 + line1_height + padding,
         'text_y2': y1 + line1_height + line2_height + 3 * padding // 2}
    ]

    for pos in positions:
        bg_x1, bg_y1 = pos['bg_x1'], pos['bg_y1']
        text_y1, text_y2 = pos['text_y1'], pos['text_y2']
        bg_x2 = bg_x1 + bg_width
        bg_y2 = bg_y1 + bg_height

        if bg_x1 < 0 or bg_y1 < 0 or bg_x2 > img_w or bg_y2 > img_h:
            continue

        overlap = False
        label_box = (bg_x1, bg_y1, bg_x2, bg_y2)
        for existing_box in existing_labels:
            if iou(label_box, existing_box) > label_iou_threshold:
                overlap = True
                break

        if not overlap:
            return pos, label_box

    default_pos = positions[0]
    default_pos['bg_y1'] = max(0, default_pos['bg_y1'])
    default_pos['text_y1'] = default_pos['bg_y1'] + line1_height + padding
    default_pos['text_y2'] = default_pos['text_y1'] + line2_height + 3 * padding // 2
    return default_pos, (
    default_pos['bg_x1'], default_pos['bg_y1'], default_pos['bg_x1'] + bg_width, default_pos['bg_y1'] + bg_height)


def adjust_zoom_ratio(class_name, phase):
    """Ajuste le zoom en fonction de la classe et de la phase"""
    zoom_params = {
        'Epinard': {'Germination': 2.5, 'Recolte': 1.5, 'Developpement_des_feuilles': 1.5},
        'Radis': {'Formation_de_la_tete': 2.0, 'Germination': 2.5, 'Recolte': 1.5, 'Developpement_des_feuilles': 1.5}
    }
    if phase == "Phase incertaine":
        return [1.5, 2.5]
    return zoom_params.get(class_name, {}).get(phase, 1.0)


def process_image(image_path, output_dir, model_dir, weights_path):
    """Traite une image pour détecter et classifier les plantes"""
    # Configuration des couleurs et chemins des modèles
    plant_colors = {
        'Epinard': (0, 255, 0),
        'Laitue_Romaine': (0, 0, 255),
        'Radis': (124, 0, 255),
        'unknown': (128, 128, 128)
    }

    models = {}
    model_paths = {
        'Epinard': os.path.join(model_dir, 'Epinard_model.keras'),
        'Laitue_Romaine': os.path.join(model_dir, 'Laitue_Romaine_model.keras'),
        'Radis': os.path.join(model_dir, 'Radis_model.keras'),
    }

    classes_phases = {
        'Epinard': ['Developpement_des_feuilles', 'Germination', 'Recolte'],
        'Laitue_Romaine': ['Developpement_des_feuilles', 'Formation_de_la_tete', 'Germination', 'Recolte'],
        'Radis': ['Developpement_des_feuilles', 'Formation_de_la_tete', 'Germination', 'Recolte'],
    }

    # Récupérer les plantes depuis Firestore
    plants_ref = db.collection('plants')
    plants = plants_ref.get()
    plant_id_mapping = {}
    for plant in plants:
        plant_data = plant.to_dict()
        nom = plant_data.get('nom')
        if nom:
            # Normaliser les noms pour correspondre à ceux utilisés dans YOLO
            normalized_nom = nom.replace(" ", "_")  # Remplace les espaces par des underscores
            if normalized_nom == "Laitue_Romaine":
                normalized_nom = "Laitue_Romaine"
            elif normalized_nom == "Epinards":
                normalized_nom = "Epinard"
            elif normalized_nom == "Radiss":
                normalized_nom = "Radis"
            if normalized_nom in plant_id_mapping:
                plant_id_mapping[normalized_nom].append(plant.id)
            else:
                plant_id_mapping[normalized_nom] = [plant.id]
    logging.info(f"plant_id_mapping: {plant_id_mapping}")  # Débogage

    # Chargement des modèles de classification
    for class_name, model_path in model_paths.items():
        try:
            models[class_name] = load_model(model_path)
            logging.info(f"Modèle chargé pour {class_name}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle {class_name} : {e}")
            models[class_name] = None

    # Chargement du modèle YOLO avec gestion sécurisée
    try:
        with torch.serialization.safe_globals([
            ultralytics.nn.tasks.DetectionModel,
            torch.nn.modules.container.Sequential,
            ultralytics.nn.modules.conv.Conv,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            ultralytics.nn.modules.block.DFL,
            ultralytics.nn.modules.head.Detect
        ]):
            yolo_model = YOLO(weights_path)
        logging.info("Modèle YOLO chargé avec succès")
    except Exception as e:
        logging.error(f"Erreur critique lors du chargement du modèle YOLO : {e}")
        raise RuntimeError(f"Échec du chargement du modèle YOLO: {str(e)}")

    # Lecture de l'image
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Impossible de lire {image_path}")
        raise ValueError("Image invalide")

    # Détection des objets avec YOLO
    try:
        results = yolo_model(image_path, conf=0.5, iou=0.5)[0]
    except Exception as e:
        logging.error(f"Erreur lors de la détection YOLO pour {image_path} : {e}")
        raise

    # Traitement des résultats
    boxes_by_class = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = results.names[cls_id]
        # Normaliser les noms détectés par YOLO (gérer les pluriels)
        if class_name == "Epinards":
            class_name = "Epinard"
        elif class_name == "Radiss":
            class_name = "Radis"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if class_name not in boxes_by_class:
            boxes_by_class[class_name] = []
        boxes_by_class[class_name].append((x1, y1, x2, y2))

    existing_labels = []
    detected_plants_dict = {}  # Dictionnaire temporaire pour regrouper par plant_id

    for class_name, boxes in boxes_by_class.items():
        fused_boxes = fuse_boxes(boxes, iou_threshold=0.6)
        color = plant_colors.get(class_name, plant_colors['unknown'])

        # Trier les boîtes pour Laitue_Romaine par position horizontale (x1)
        if class_name == 'Laitue_Romaine':
            fused_boxes = sorted(fused_boxes, key=lambda box: box[0])  # De gauche à droite

        available_plant_ids = plant_id_mapping.get(class_name, [])  # Copie des IDs disponibles
        plant_id_counts = {pid: 0 for pid in available_plant_ids}  # Suivre l'utilisation des plant_id
        for idx, (x1, y1, x2, y2) in enumerate(fused_boxes):
            # Dessin du rectangle autour de la plante
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            predicted_phases = []  # Liste pour accumuler toutes les phases prédites
            if class_name in models and models[class_name] is not None:
                try:
                    # Initialisation du zoom
                    zoom_ratio = 1.0
                    center_x = (x1 + x2) // 2
                    center_y = y1 + (y2 - y1) // 4
                    box_w = x2 - x1
                    box_h = y2 - y1

                    # Premier passage de prédiction
                    new_w = int(box_w * zoom_ratio)
                    new_h = int(box_h * zoom_ratio)
                    new_x1 = max(0, center_x - new_w // 2)
                    new_y1 = max(0, center_y - new_h // 2)
                    new_x2 = min(img.shape[1], center_x + new_w // 2)
                    new_y2 = min(img.shape[0], center_y + new_h // 2)
                    cropped = img[new_y1:new_y2, new_x1:new_x2]

                    if cropped.size == 0:
                        predicted_phases.append("Erreur de cropping")
                        continue

                    resized = cv2.resize(cropped, (224, 224))
                    img_array = image.img_to_array(resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    predictions = models[class_name].predict(img_array, verbose=0)[0]
                    predicted_idx = np.argmax(predictions)
                    confidence = predictions[predicted_idx]
                    min_confidence = 0.5

                    if confidence >= min_confidence:
                        predicted_phases.append(classes_phases[class_name][predicted_idx])
                    else:
                        predicted_phases.append("Phase incertaine")

                    # Ajustement du zoom si nécessaire
                    zoom_ratios = adjust_zoom_ratio(class_name, predicted_phases[0])
                    if isinstance(zoom_ratios, list):
                        best_phase = predicted_phases[0]
                        best_confidence = confidence
                        for zoom in zoom_ratios:
                            new_w = int(box_w * zoom)
                            new_h = int(box_h * zoom)
                            new_x1 = max(0, center_x - new_w // 2)
                            new_y1 = max(0, center_y - new_h // 2)
                            new_x2 = min(img.shape[1], center_x + new_w // 2)
                            new_y2 = min(img.shape[0], center_y + new_h // 2)
                            cropped = img[new_y1:new_y2, new_x1:new_x2]

                            if cropped.size == 0:
                                continue

                            resized = cv2.resize(cropped, (224, 224))
                            img_array = image.img_to_array(resized)
                            img_array = np.expand_dims(img_array, axis=0)
                            img_array = preprocess_input(img_array)
                            predictions = models[class_name].predict(img_array, verbose=0)[0]
                            new_idx = np.argmax(predictions)
                            new_confidence = predictions[new_idx]

                            if new_confidence > best_confidence:
                                best_confidence = new_confidence
                                best_phase = classes_phases[class_name][new_idx]
                                predicted_phases.append(best_phase)  # Ajouter la meilleure phase

                    else:
                        zoom_ratio = zoom_ratios
                        # Nouvelle tentative avec le zoom ajusté
                        new_w = int(box_w * zoom_ratio)
                        new_h = int(box_h * zoom_ratio)
                        new_x1 = max(0, center_x - new_w // 2)
                        new_y1 = max(0, center_y - new_h // 2)
                        new_x2 = min(img.shape[1], center_x + new_w // 2)
                        new_y2 = min(img.shape[0], center_y + new_h // 2)
                        cropped = img[new_y1:new_y2, new_x1:new_x2]

                        if cropped.size == 0:
                            continue

                        resized = cv2.resize(cropped, (224, 224))
                        img_array = image.img_to_array(resized)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = preprocess_input(img_array)
                        predictions = models[class_name].predict(img_array, verbose=0)[0]
                        predicted_idx = np.argmax(predictions)
                        confidence = predictions[predicted_idx]

                        if confidence >= min_confidence:
                            predicted_phases.append(classes_phases[class_name][predicted_idx])
                        else:
                            predicted_phases.append("Phase incertaine")

                except Exception as e:
                    logging.error(f"Erreur de prédiction pour {class_name} dans {image_path} : {e}")
                    predicted_phases.append("Erreur de prédiction")

            # Associer un plant_id
            plant_id = None
            if class_name in plant_id_mapping and available_plant_ids:
                if class_name == 'Laitue_Romaine':
                    # Associer en fonction de la position : gauche (avant) -> premier plant_id, droite (arrière) -> second plant_id
                    if idx < len(available_plant_ids):
                        plant_id = available_plant_ids[idx]
                    else:
                        plant_id = available_plant_ids[idx % len(available_plant_ids)]
                        logging.warning(f"Plus de {class_name} détectées que dans la collection plants, réutilisation de plant_id {plant_id}")
                else:
                    if available_plant_ids:
                        plant_id = min(plant_id_counts, key=plant_id_counts.get)
                        plant_id_counts[plant_id] += 1
                    else:
                        logging.warning(f"Aucun plant_id disponible pour {class_name}")
            else:
                logging.warning(f"Aucun plant_id correspondant trouvé pour {class_name}")

            # Ajouter ou mettre à jour la détection dans le dictionnaire temporaire
            if plant_id:
                if plant_id not in detected_plants_dict:
                    detected_plants_dict[plant_id] = {
                        'plant_id': plant_id,
                        'plant_name': class_name,
                        'predicted_phases': []  # Liste pour conserver tous les doublons
                    }
                detected_plants_dict[plant_id]['predicted_phases'].extend(predicted_phases)

            # Préparation du texte à afficher (utiliser la dernière phase prédite pour l'étiquette)
            line1 = f"{class_name}"
            line2 = predicted_phases[-1] if predicted_phases else "Modèle non chargé"

            # Paramètres d'affichage
            font_scale = 1.5
            thickness_border = 6
            thickness_text = 3
            padding = 10

            # Positionnement de l'étiquette
            pos, label_box = get_label_position(
                img, (x1, y1, x2, y2), line1, line2,
                font_scale, thickness_text, padding, existing_labels
            )
            existing_labels.append(label_box)

            # Calcul des dimensions de l'arrière-plan
            bg_width = max(
                cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)[0][0],
                cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)[0][0]
            ) + 2 * padding

            bg_height = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)[0][1] + \
                        cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)[0][1] + 3 * padding

            # Dessin de l'étiquette
            cv2.rectangle(img, (pos['bg_x1'], pos['bg_y1']),
                          (pos['bg_x1'] + bg_width, pos['bg_y1'] + bg_height),
                          color, -1)

            # Texte avec contour
            cv2.putText(img, line1, (pos['bg_x1'] + padding, pos['text_y1']),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_border)
            cv2.putText(img, line1, (pos['bg_x1'] + padding, pos['text_y1']),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_text)

            cv2.putText(img, line2, (pos['bg_x1'] + padding, pos['text_y2']),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_border)
            cv2.putText(img, line2, (pos['bg_x1'] + padding, pos['text_y2']),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_text)

    # Convertir le dictionnaire en liste pour Firestore
    detected_plants = [data for data in detected_plants_dict.values()]

    # Sauvegarde de l'image annotée
    output_path = os.path.join(output_dir, f"annotée_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, img)
    logging.info(f"Image traitée enregistrée dans {output_path}")

    # Uploader l'image sur Cloudinary avec compression automatique
    try:
        upload_result = cloudinary.uploader.upload(
            output_path,
            folder="plant_predictions",
            resource_type="image",
            quality="auto"
        )
        image_url = upload_result['secure_url']
        logging.info(f"Image uploadée sur Cloudinary : {image_url}")

        # Enregistrer un seul document dans Firestore
        doc_ref = db.collection('plant_predictions').document()
        doc_ref.set({
            'image_url': image_url,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'detected_plants': detected_plants
        })
        logging.info(f"Résultat enregistré dans Firestore avec ID : {doc_ref.id}")

        # Supprimer les fichiers temporaires
        os.remove(output_path)
        logging.info(f"Fichier temporaire supprimé : {output_path}")
        os.remove(image_path)
        logging.info(f"Image originale supprimée : {image_path}")

        # Retourner l'URL Cloudinary
        return image_url

    except Exception as e:
        logging.error(f"Erreur lors de l'upload sur Cloudinary ou l'enregistrement dans Firestore : {e}")
        # En cas d'erreur, retourner le chemin local
        return output_path


def get_sorted_predictions():
    """Récupère les prédictions de Firestore triées par timestamp (du plus récent au plus ancien)"""
    try:
        # Récupérer les documents de plant_predictions, triés par timestamp en ordre décroissant
        predictions_ref = db.collection('plant_predictions').order_by('timestamp', direction=firestore.Query.DESCENDING)
        predictions = predictions_ref.get()

        # Convertir les documents en une liste de dictionnaires
        sorted_predictions = [prediction.to_dict() for prediction in predictions]

        logging.info(f"Récupéré {len(sorted_predictions)} prédictions triées par date")
        return sorted_predictions

    except Exception as e:
        logging.error(f"Erreur lors de la récupération des prédictions triées : {e}")
        return []