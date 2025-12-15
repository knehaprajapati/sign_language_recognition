from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import atexit
import threading

app = Flask(__name__)

# Load trained YOLOv8 pose model
model_path = "model/best (1) (1).pt"
model = YOLO(model_path)

print(" Model ke andar ye classes hain:", model.names)

# ✅ Camera ko yahan ek hi baar open karo
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ✅ Exit pe camera release ho jaaye
@atexit.register
def cleanup():
    if cap.isOpened():
        cap.release()
        print("🧹 Camera released on exit.")

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    print(" Starting webcam stream...")

    if not cap.isOpened():
        print(" Failed to open camera")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print(" Failed to read frame")
            continue

        results = model(frame)

        if results[0].probs is not None:
            class_id = int(results[0].probs.top1)
            conf = float(results[0].probs.top1conf)
            activity = model.names[class_id]
            label = f"{activity} ({conf*100:.1f}%)"
        else:
            label = "Detecting..."

        annotated_frame = results[0].plot()

        cv2.putText(
            annotated_frame,
            f"Activity: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No image uploaded"})

    try:
        image = Image.open(request.files['file'].stream).convert("RGB")
        image = np.array(image)

        results = model(image)

        predictions = []
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                predictions.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                })

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Start Flask app in a separate thread
    threading.Thread(target=lambda: app.run(debug=True, use_reloader=False)).start()
