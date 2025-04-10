import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    attention_state = "Idle"
    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2

        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        if abs(cx - frame_center_x) < 100 and abs(cy - frame_center_y) < 100:
            attention_state = "Focused"
        else:
            attention_state = "Distracted"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Attention: {attention_state}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Attention Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
