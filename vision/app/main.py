import cv2

from ai.camera import Camera
from ai.face_detector import FaceDetector


camera = Camera()

detector = FaceDetector()


while True:

    ret, frame = camera.read()

    if not ret:
        break

    faces = detector.detect(frame)

    for face in faces:

        box = face.bbox.astype(int)

        x1, y1, x2, y2 = box

        confidence = face.det_score

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("SmartVision AI - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()