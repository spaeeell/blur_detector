


import cv2
import numpy as np
import json

from pathlib import Path


IMAGE_DIR = Path("images")
CONFIG_FILE = Path("quality_config.json")

FACE_SIZE = (320, 320)

WEIGHTS = {
    "laplacian": 0.40,
    "tenengrad": 0.30,
    "fft": 0.20,
    "edge_density": 0.10,
}


CASCADE_PATH = r"C:\Users\arsen\opencv_data\haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError("Не удалось загрузить модель распознавания лица")


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        # Если лицо не нашли, временно берём центральную часть.
        h, w = image.shape[:2]

        x1 = int(w * 0.10)
        x2 = int(w * 0.90)

        y1 = int(h * 0.10)
        y2 = int(h * 0.90)

        return image[y1:y2, x1:x2], False

    # Если лиц несколько — берём самое большое.
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Немного пространства вокруг лица.
    padding = int(max(w, h) * 0.15)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    face = image[y1:y2, x1:x2]

    return face, True


def calculate_features(image):
    face, face_found = detect_face(image)

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        FACE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    # 1. Laplacian
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # 2. Tenengrad / Sobel
    gx = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
    )

    tenengrad = np.mean(
        gx * gx + gy * gy
    )

    # 3. FFT — количество высоких частот
    fft = np.fft.fft2(
        gray.astype(np.float32)
    )

    fft = np.fft.fftshift(fft)

    power = np.abs(fft) ** 2

    height, width = gray.shape

    cy = height // 2
    cx = width // 2

    yy, xx = np.ogrid[
        :height,
        :width
    ]

    radius = 25

    mask = (
        (xx - cx) ** 2
        + (yy - cy) ** 2
        >= radius ** 2
    )

    high_frequency = (
        power[mask].sum()
        / (power.sum() + 1e-12)
    )

    # 4. Количество чётких границ
    edges = cv2.Canny(
        gray,
        80,
        160,
    )

    edge_density = (
        np.count_nonzero(edges)
        / edges.size
    )

    # Дополнительная информация
    brightness = gray.mean()
    contrast = gray.std()

    return {
        "laplacian": float(laplacian),
        "tenengrad": float(tenengrad),
        "fft": float(high_frequency),
        "edge_density": float(edge_density),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "face_found": face_found,
    }


def normalize(value, minimum, maximum):
    if maximum == minimum:
        return 0

    value = (
        (value - minimum)
        / (maximum - minimum)
    )

    return float(
        np.clip(value, 0, 1)
    )


def calculate_score(features, ranges):
    score = 0

    for name, weight in WEIGHTS.items():

        minimum = ranges[name]["min"]
        maximum = ranges[name]["max"]

        normalized = normalize(
            features[name],
            minimum,
            maximum,
        )

        score += normalized * weight

    return score * 100


def find_best_threshold(results):
    labelled = [
        item
        for item in results
        if item["label"] is not None
    ]

    scores = sorted(
        set(
            item["score"]
            for item in labelled
        )
    )

    candidates = []

    for i in range(len(scores) - 1):
        candidates.append(
            (scores[i] + scores[i + 1])
            / 2
        )

    best_threshold = 50
    best_accuracy = 0

    for threshold in candidates:

        correct = 0

        for item in labelled:

            predicted_good = (
                item["score"]
                >= threshold
            )

            real_good = (
                item["label"]
                == "good"
            )

            if predicted_good == real_good:
                correct += 1

        accuracy = (
            correct
            / len(labelled)
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def main():
    results = []

    files = sorted(
        list(IMAGE_DIR.glob("*.jpg"))
        + list(IMAGE_DIR.glob("*.jpeg"))
        + list(IMAGE_DIR.glob("*.png"))
    )

    if not files:
        print("В папке images нет фотографий.")
        return

    print("Анализ фотографий...\n")

    for path in files:

        image = read_image(path)

        if image is None:
            print(
                f"Не удалось открыть {path.name}"
            )
            continue

        features = calculate_features(
            image
        )

        label = None

        name = path.stem.lower()

        if "_good" in name:
            label = "good"

        elif "_bad" in name:
            label = "bad"

        results.append({
            "filename": path.name,
            "label": label,
            "features": features,
        })

    labelled = [
        item
        for item in results
        if item["label"] is not None
    ]

    if len(labelled) < 2:
        print(
            "Для калибровки нужны "
            "файлы _good и _bad."
        )
        return

    ranges = {}

    for feature_name in WEIGHTS:

        values = [
            item["features"][feature_name]
            for item in labelled
        ]

        # Берём 5 и 95 процентиль,
        # чтобы единичные выбросы
        # не ломали шкалу.
        minimum = np.percentile(
            values,
            5
        )

        maximum = np.percentile(
            values,
            95
        )

        ranges[feature_name] = {
            "min": float(minimum),
            "max": float(maximum),
        }

    for item in results:

        item["score"] = calculate_score(
            item["features"],
            ranges,
        )

    threshold, accuracy = (
        find_best_threshold(results)
    )

    print(
        f"Автоматический порог: "
        f"{threshold:.2f}"
    )

    print(
        f"Точность на тестовой выборке: "
        f"{accuracy * 100:.1f}%"
    )

    print("\nРезультаты:\n")

    for item in results:

        score = item["score"]

        if score >= threshold:
            status = "GOOD"
        else:
            status = "BAD"

        face_status = (
            "face"
            if item["features"]["face_found"]
            else "face not found"
        )

        print(
            f"{item['filename']:20} "
            f"{score:6.2f} "
            f"-> {status:4} "
            f"({face_status})"
        )

    config = {
        "threshold": threshold,
        "weights": WEIGHTS,
        "ranges": ranges,
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    print(
        "\nНастройки сохранены "
        "в quality_config.json"
    )


if __name__ == "__main__":
    main()



#import cv2
#import os


#def check_blur(image_path):
#    image = cv2.imread(image_path)

#    if image is None:
#        return None

#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#    score = cv2.Laplacian(
#        gray,
#        cv2.CV_64F
#    ).var()

#    return score


#folder = "images"

#for filename in os.listdir(folder):
#    path = os.path.join(folder, filename)
#    score = check_blur(path)

#    if score is not None:
#        print(f"{filename}: {score:.2f}")