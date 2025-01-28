from flask import Flask, render_template, jsonify, request, Response
import numpy as np
from yolo_dnn import run_module
import cv2
import base64


# img_path = r"D:\Magistratura_3_semestr\id_446_value_271_545.jpg"

# img_result = run_module(img_path)
# cv2.imwrite(
#     r"C:\MyPythonProjects\myflaskproject\app\static\images\img1_pred.jpg", img_result
# )


# Flaskapp reaslization
app = Flask(__name__)

# наработка за 13.12.2024 - отобразить видеопоток с кнопкой фотографии, и вернуть снизу отредактированную картинку


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    # Получение изображения от клиента
    data = request.json["image"]
    # Декодирование изображения из base64
    image_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    result = run_module(image)
    # Преобразование в черно-белый формат
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Конвертация обратно в base64 для отправки клиенту
    _, buffer = cv2.imencode(".jpg", result)
    gray_image_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": f"data:image/jpeg;base64,{gray_image_base64}"})


if __name__ == "__main__":
    app.run(debug=True)
