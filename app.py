# from flask import Flask, request, render_template
# from ultralytics import YOLO
# import os

# app = Flask(__name__)

# model = YOLO("best.pt")

# UPLOAD_FOLDER = "static/uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         file = request.files["image"]
#         path = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(path)

#         results = model(path, save=True)

#         return render_template("index.html", image=path)

#     return render_template("index.html", image=None)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

# Load model
model = YOLO("best.pt")

# Folders
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


# Home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        if file.filename == "":
            return "No file selected"

        # Save uploaded image
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(upload_path)

        # Run prediction
        results = model(upload_path)

        # Get result image with boxes
        result_img = results[0].plot()

        # Save output image
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], file.filename)
        cv2.imwrite(output_path, result_img)

        return render_template(
            "index.html",
            uploaded_image=upload_path,
            output_image=output_path
        )

    return render_template("index.html", uploaded_image=None, output_image=None)


# Run app
if __name__ == "__main__":
    app.run(debug=True)