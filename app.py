
from flask import Flask, request, render_template
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

# ✅ Load model
model = YOLO("best.pt")

# Folders
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

        # ✅ File validation
        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]

        if file.filename == "":
            return "No file selected"

        try:
            # ✅ Save uploaded image
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            upload_path = upload_path.replace("\\", "/")
            file.save(upload_path)

            # ✅ Run YOLO prediction
            results = model(upload_path)

            # ✅ Read image
            img = cv2.imread(upload_path)

            danger_count = 0
            normal_count = 0

            # ✅ Loop detections
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = model.names[cls]

                # ✅ CORRECT CONFIDENCE LOGIC
                if conf < 0.6:
                    color = (0, 0, 255)  # 🔴 RED = danger
                    danger_count += 1
                else:
                    color = (255, 0, 0)  # 🔵 BLUE = normal
                    normal_count += 1

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # Label text
                text = f"{label} {conf:.2f}"

                # Background for text
                cv2.rectangle(img, (x1, y1 - 30), (x1 + 160, y1), color, -1)

                # Put label text
                cv2.putText(img, text, (x1 + 5, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # ✅ Save output image
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], file.filename)
            output_path = output_path.replace("\\", "/")
            cv2.imwrite(output_path, img)

            # ✅ Prediction result
            if danger_count > 0:
                prediction = f"⚠️ Danger: {danger_count} | Normal: {normal_count}"
            else:
                prediction = f"✅ Safe: {normal_count} Vegetation Detected"

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


# ✅ Run (Render compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)