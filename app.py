# from flask import Flask, request, render_template
# from ultralytics import YOLO
# import os
# import cv2

# app = Flask(__name__)

# # Load YOLO model
# model = YOLO("best.pt")

# # Folders
# UPLOAD_FOLDER = "static/uploads"
# OUTPUT_FOLDER = "static/outputs"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":

#         # Check file
#         if "image" not in request.files:
#             return "No file uploaded"

#         file = request.files["image"]

#         if file.filename == "":
#             return "No file selected"

#         # Save uploaded image
#         upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#         file.save(upload_path)

#         # Run YOLO prediction
#         results = model(upload_path)

#         # Draw bounding boxes
#         result_img = results[0].plot()

#         # Save output image
#         output_path = os.path.join(app.config["OUTPUT_FOLDER"], file.filename)
#         cv2.imwrite(output_path, result_img)

#         return render_template(
#             "index.html",
#             uploaded_image="/" + upload_path,
#             output_image="/" + output_path,
#             prediction="Vegetation Detected 🌿"
#         )

#     return render_template(
#         "index.html",
#         uploaded_image=None,
#         output_image=None,
#         prediction=None
#     )


# if __name__ == "__main__":
#     app.run(debug=True)










from flask import Flask, request, render_template
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

# ✅ Load model once (important for performance)
model = YOLO("best.pt")

# ✅ Folder setup
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image = None
    output_image = None
    prediction = None

    if request.method == "POST":

        # ✅ Check file exists
        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]

        if file.filename == "":
            return "No file selected"

        try:
            # ✅ Save uploaded file
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(upload_path)

            # ✅ Run YOLO prediction
            results = model(upload_path)

            # ✅ Draw bounding boxes
            result_img = results[0].plot()

            # ✅ Save output image
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], file.filename)
            cv2.imwrite(output_path, result_img)

            # ✅ Simple prediction text
            if len(results[0].boxes) > 0:
                prediction = f"{len(results[0].boxes)} Vegetation Detected 🌿"
            else:
                prediction = "No Vegetation Detected ❌"

            uploaded_image = "/" + upload_path
            output_image = "/" + output_path

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template(
        "index.html",
        uploaded_image=uploaded_image,
        output_image=output_image,
        prediction=prediction
    )


# ✅ Render-compatible run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)