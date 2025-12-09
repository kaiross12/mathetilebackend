from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import uuid
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def rotate_bound(image, angle):
    """
    Bir görüntüyü belirtilen açıda döndürür ve
    kırpılmaması için tuval boyutunu genişletir.
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Döndürme matrisi
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Yeni tuval boyutlarını hesapla
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Matrisi merkeze göre ayarla
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Döndürme işlemi (Kenar boşluklarını açık gri yapıyoruz)
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(200, 200, 200))

def non_max_suppression_indices(boxes, scores=None, overlapThresh=0.3):
    """
    NMS: Çakışan kutular arasından en iyi indeksleri döndürür.
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype("float")
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    if scores is not None and len(scores) == len(boxes):
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    pick = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        
        remove_idxs = np.where(overlap > overlapThresh)[0]
        idxs = np.delete(idxs, np.concatenate(([last], remove_idxs)))

    return pick

def calculate_symmetry(img):
    """
    Basit simetri skoru hesaplar.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    half = w // 2
    if half == 0:
        return 0

    left = gray[:, :half].astype(np.float32)
    right = gray[:, w - half:].astype(np.float32)
    right_flipped = np.fliplr(right)

    min_w = min(left.shape[1], right_flipped.shape[1])
    left_c = left[:, :min_w]
    right_c = right_flipped[:, :min_w]

    diff = np.abs(left_c - right_c)
    mean_diff = np.mean(diff)
    similarity = max(0.0, 1.0 - (mean_diff / 255.0))
    return int(similarity * 100)

@app.post("/analyze-template")
def analyze_template():
    try:
        # 1. Dosya Kontrolleri
        if "file" not in request.files:
            return jsonify({"error": "file alanı yok."}), 400
        file = request.files["file"]
        
        filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
        path = os.path.join(UPLOAD_DIR, filename)
        file.save(path)

        img = cv2.imread(path)
        if img is None:
            return jsonify({"error": "Resim okunamadı."}), 400

        ih, iw = img.shape[:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur işlemi (Benzerlik araması için)
        img_gray_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # 2. Parametreleri Al
        try:
            roi_x = float(request.form.get("x", 0))
            roi_y = float(request.form.get("y", 0))
            roi_w = float(request.form.get("w", 0))
            roi_h = float(request.form.get("h", 0))
            threshold = float(request.form.get("threshold", 0.60))
        except ValueError:
            return jsonify({"error": "Parametre hatası."}), 400

        # ROI Hesaplama
        x1 = int(roi_x * iw)
        y1 = int(roi_y * ih)
        w_px = int(roi_w * iw)
        h_px = int(roi_h * ih)
        
        x1 = max(0, min(x1, iw - 1))
        y1 = max(0, min(y1, ih - 1))
        w_px = max(1, min(w_px, iw - x1))
        h_px = max(1, min(h_px, ih - y1))

        # Ana Şablonu Kes
        base_template = img_gray_blurred[y1:y1+h_px, x1:x1+w_px]
        if base_template.shape[0] < 2 or base_template.shape[1] < 2:
            return jsonify({"error": "Seçim çok küçük."}), 400

        # -----------------------------------------------------------
        # AÇI TARAMASI
        # step = 20 -> Her 20 derecede bir resmi çevirip arar.
        # -----------------------------------------------------------
        step = 20
        templates = []
        
        for angle in range(0, 360, step):
            rotated_img = rotate_bound(base_template, angle)
            templates.append({"img": rotated_img, "angle": angle})

        all_rects = []
        all_scores = []

        # Her açı için tarama
        for temp_data in templates:
            t_img = temp_data["img"]
            th, tw = t_img.shape[:2]

            # Şablon resimden büyük olamaz
            if th > ih or tw > iw:
                continue

            res = cv2.matchTemplate(img_gray_blurred, t_img, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            for (py, px) in zip(*loc):
                all_rects.append([int(px), int(py), int(px + tw), int(py + th)])
                all_scores.append(float(res[py, px]))

        # Formatlama ve NMS
        all_rects = np.array(all_rects, dtype=np.int32) if len(all_rects) > 0 else np.empty((0,4), dtype=np.int32)
        all_scores = np.array(all_scores, dtype=np.float32) if len(all_scores) > 0 else np.array([], dtype=np.float32)

        if all_rects.shape[0] == 0:
            return jsonify({
                "symmetry": calculate_symmetry(img),
                "motifs": [],
                "count": 0
            })

        # NMS ile en iyi çakışmaları seç
        picks = non_max_suppression_indices(all_rects, scores=all_scores, overlapThresh=0.3)
        picked_boxes = all_rects[picks]
        picked_scores = all_scores[picks]

        results = []
        for (bx, by, bx2, by2), conf in zip(picked_boxes, picked_scores):
            bw = bx2 - bx
            bh = by2 - by
            results.append({
                "x": float(bx) / iw,
                "y": float(by) / ih,
                "w": float(bw) / iw,
                "h": float(bh) / ih,
                "confidence": float(round(float(conf), 3)),
                "label": "Motif"
            })

        symmetry_score = calculate_symmetry(img)
        
        # İsterseniz işlemi biten dosyayı silebilirsiniz:
        # os.remove(path) 

        return jsonify({
            "symmetry": symmetry_score,
            "motifs": results,
            "count": len(results)
        })
        
    except Exception as e:
        app.logger.exception("Hata")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)