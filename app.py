
# from flask import Flask, request, render_template
# from ultralytics import YOLO
# import os
# import cv2

# app = Flask(__name__)

# # ✅ Load model
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
#     uploaded_image = None
#     output_image = None
#     prediction = None

#     if request.method == "POST":

#         # ✅ File validation
#         if "image" not in request.files:
#             return "No file uploaded"

#         file = request.files["image"]

#         if file.filename == "":
#             return "No file selected"

#         try:
#             # ✅ Save uploaded image
#             upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#             upload_path = upload_path.replace("\\", "/")
#             file.save(upload_path)

#             # ✅ Run YOLO prediction
#             results = model(upload_path)

#             # ✅ Read image
#             img = cv2.imread(upload_path)

#             danger_count = 0
#             normal_count = 0

#             # ✅ Loop detections
#             for box in results[0].boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])
#                 cls = int(box.cls[0])

#                 label = model.names[cls]

#                 # ✅ CORRECT CONFIDENCE LOGIC
#                 if conf < 0.6:
#                     color = (0, 0, 255)  # 🔴 RED = danger
#                     danger_count += 1
#                 else:
#                     color = (255, 0, 0)  # 🔵 BLUE = normal
#                     normal_count += 1

#                 # Draw bounding box
#                 cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

#                 # Label text
#                 text = f"{label} {conf:.2f}"

#                 # Background for text
#                 cv2.rectangle(img, (x1, y1 - 30), (x1 + 160, y1), color, -1)

#                 # Put label text
#                 cv2.putText(img, text, (x1 + 5, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#             # ✅ Save output image
#             output_path = os.path.join(app.config["OUTPUT_FOLDER"], file.filename)
#             output_path = output_path.replace("\\", "/")
#             cv2.imwrite(output_path, img)

#             # ✅ Prediction result
#             if danger_count > 0:
#                 prediction = f"⚠️ Danger: {danger_count} | Normal: {normal_count}"
#             else:
#                 prediction = f"✅ Safe: {normal_count} Vegetation Detected"

#             uploaded_image = "/" + upload_path
#             output_image = "/" + output_path

#         except Exception as e:
#             return f"Error: {str(e)}"

#     return render_template(
#         "index.html",
#         uploaded_image=uploaded_image,
#         output_image=output_image,
#         prediction=prediction
#     )


# # ✅ Run (Render compatible)
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port)




import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

st.set_page_config(page_title="Vegetation Detection", layout="centered")

st.title("🌿 Vegetation Detection using YOLO")
st.write("Upload an image to detect vegetation and analyze safety.")

# ✅ Model path
MODEL_PATH = "best.pt"

# ✅ Download model FIRST (before loading)
MODEL_URL = "YOUR_MODEL_LINK"   # 🔥 replace this

if not os.path.exists(MODEL_PATH):
    st.info("⬇️ Downloading model... please wait")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ✅ Load model AFTER download
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ✅ Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("📷 Uploaded Image")
    st.image(image, use_column_width=True)

    # YOLO prediction
    results = model(img)

    danger_count = 0
    normal_count = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        label = model.names[cls]

        # ✅ Confidence logic
        if conf < 0.6:
            color = (0, 0, 255)
            danger_count += 1
        else:
            color = (255, 0, 0)
            normal_count += 1

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        text = f"{label} {conf:.2f}"
        cv2.rectangle(img, (x1, y1 - 30), (x1 + 160, y1), color, -1)
        cv2.putText(img, text, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show result
    st.subheader("🔍 Detection Result")
    st.image(img, channels="BGR", use_column_width=True)

    if danger_count > 0:
        st.error(f"⚠️ Danger: {danger_count} | Normal: {normal_count}")
    else:
        st.success(f"✅ Safe: {normal_count} Vegetation Detected")