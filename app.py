from flask import Flask,jsonify, render_template, Response, request
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import os
app = Flask(__name__)

model = YOLO("weights/best.pt")
class_list = ['class1', 'class2', 'class3', 'class4']
tracker = DeepSort(max_age=30)
counted_ids = defaultdict(set)
total_counts = defaultdict(int)

import numpy as np  # Make sure this is imported at the top

def generate_frames(source=0):
    cap = cv2.VideoCapture(source)
    is_file = isinstance(source, str)  # True if using a video file

    while True:
        success, frame = cap.read()
        if not success:
            if is_file:
                # Create a blank frame with "Video Ended" message
                frame = np.zeros((400, 800, 3), dtype=np.uint8)
                cv2.putText(frame, "Video Ended", (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                break
            else:
                continue  # if webcam fails briefly, try again

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id >= len(class_list):
                continue
            class_name = class_list[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            class_name = track.get_det_class()

            if track_id not in counted_ids[class_name]:
                counted_ids[class_name].add(track_id)
                total_counts[class_name] += 1

            cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ID:{track_id}", (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y_offset = 30
        for cls, count in total_counts.items():
            cv2.putText(frame, f"{cls}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            y_offset += 30

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html', show_video=False)

@app.route('/video')
def video():
    source = request.args.get('source', 'file')
    if source == 'webcam':
        video_source = 0
    else:
        video_source = "static/testvideo2.mp4"

    return Response(generate_frames(video_source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/counts')
def get_counts():
    return jsonify(total_counts)
if __name__ == '__main__':
    port=int(os.environ.get("PORT",5000))
    app.run(host='0.0.0.0', port=port)

