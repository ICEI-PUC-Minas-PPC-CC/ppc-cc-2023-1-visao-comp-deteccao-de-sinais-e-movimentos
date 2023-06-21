import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Coordenadas do quadrado
x_min = 0.1
x_max = 0.3
y_min = 0.3
y_max = 0.7

# Posição relativa dos dedos para os gestos A, B, C e D
gesture_a = {
    'thumb_tip': (0.25, 0.45),
    'index_tip': (0.22, 0.53),
    'middle_tip': (0.20, 0.55),
    'ring_tip': (0.19, 0.57),
    'pinky_tip': (0.17, 0.58)
}

gesture_b = {
    'thumb_tip': (0.20, 0.53),
    'index_tip': (0.24, 0.37),
    'middle_tip': (0.22, 0.35),
    'ring_tip': (0.21, 0.37),
    'pinky_tip': (0.18, 0.42)
}

gesture_c = {
    'thumb_tip': (0.25, 0.51),
    'index_tip': (0.22, 0.42),
    'middle_tip': (0.22, 0.42),
    'ring_tip': (0.21, 0.43),
    'pinky_tip': (0.19, 0.44)
}

gesture_d = {
    'thumb_tip': (0.25, 0.51),
    'index_tip': (0.20, 0.37),
    'middle_tip': (0.24, 0.49),
    'ring_tip': (0.23, 0.51),
    'pinky_tip': (0.21, 0.50)
}

def recognize_gesture(gesture):
    print(f"Gesto '{gesture}' reconhecido")

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    start_time = time.time()  # Definindo o tempo inicial

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Não foi possível ler o quadro da câmera.")
            break

        image = cv2.flip(image, 1)

        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                # Desenha o quadrado na imagem
                image_height, image_width, _ = image.shape
                x_min_pixel = int(x_min * image_width)
                x_max_pixel = int(x_max * image_width)
                y_min_pixel = int(y_min * image_height)
                y_max_pixel = int(y_max * image_height)
                cv2.rectangle(image, (x_min_pixel, y_min_pixel), (x_max_pixel, y_max_pixel), (0, 255, 0), 2)

                # Obtém a posição dos dedos da mão
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Verifica se a mão está dentro do quadrado
                if (
                    x_min <= thumb_tip.x <= x_max and
                    x_min <= index_tip.x <= x_max and
                    x_min <= middle_tip.x <= x_max and
                    x_min <= ring_tip.x <= x_max and
                    x_min <= pinky_tip.x <= x_max and
                    y_min <= thumb_tip.y <= y_max and
                    y_min <= index_tip.y <= y_max and
                    y_min <= middle_tip.y <= y_max and
                    y_min <= ring_tip.y <= y_max and
                    y_min <= pinky_tip.y <= y_max
                ):
                    # Verifica o gesto A
                    if (
                        gesture_a['thumb_tip'][0] - 0.02 <= thumb_tip.x <= gesture_a['thumb_tip'][0] + 0.02 and
                        gesture_a['index_tip'][0] - 0.02 <= index_tip.x <= gesture_a['index_tip'][0] + 0.02 and
                        gesture_a['middle_tip'][0] - 0.02 <= middle_tip.x <= gesture_a['middle_tip'][0] + 0.02 and
                        gesture_a['ring_tip'][0] - 0.02 <= ring_tip.x <= gesture_a['ring_tip'][0] + 0.02 and
                        gesture_a['pinky_tip'][0] - 0.02 <= pinky_tip.x <= gesture_a['pinky_tip'][0] + 0.02 and
                        gesture_a['thumb_tip'][1] - 0.02 <= thumb_tip.y <= gesture_a['thumb_tip'][1] + 0.02 and
                        gesture_a['index_tip'][1] - 0.02 <= index_tip.y <= gesture_a['index_tip'][1] + 0.02 and
                        gesture_a['middle_tip'][1] - 0.02 <= middle_tip.y <= gesture_a['middle_tip'][1] + 0.02 and
                        gesture_a['ring_tip'][1] - 0.02 <= ring_tip.y <= gesture_a['ring_tip'][1] + 0.02 and
                        gesture_a['pinky_tip'][1] - 0.02 <= pinky_tip.y <= gesture_a['pinky_tip'][1] + 0.02
                    ):
                        recognize_gesture('A')

                    # Verifica o gesto B
                    elif (
                        gesture_b['thumb_tip'][0] - 0.02 <= thumb_tip.x <= gesture_b['thumb_tip'][0] + 0.02 and
                        gesture_b['index_tip'][0] - 0.02 <= index_tip.x <= gesture_b['index_tip'][0] + 0.02 and
                        gesture_b['middle_tip'][0] - 0.02 <= middle_tip.x <= gesture_b['middle_tip'][0] + 0.02 and
                        gesture_b['ring_tip'][0] - 0.02 <= ring_tip.x <= gesture_b['ring_tip'][0] + 0.02 and
                        gesture_b['pinky_tip'][0] - 0.02 <= pinky_tip.x <= gesture_b['pinky_tip'][0] + 0.02 and
                        gesture_b['thumb_tip'][1] - 0.02 <= thumb_tip.y <= gesture_b['thumb_tip'][1] + 0.02 and
                        gesture_b['index_tip'][1] - 0.02 <= index_tip.y <= gesture_b['index_tip'][1] + 0.02 and
                        gesture_b['middle_tip'][1] - 0.02 <= middle_tip.y <= gesture_b['middle_tip'][1] + 0.02 and
                        gesture_b['ring_tip'][1] - 0.02 <= ring_tip.y <= gesture_b['ring_tip'][1] + 0.02 and
                        gesture_b['pinky_tip'][1] - 0.02 <= pinky_tip.y <= gesture_b['pinky_tip'][1] + 0.02
                    ):
                        recognize_gesture('B')

                    # Verifica o gesto C
                    elif (
                        gesture_c['thumb_tip'][0] - 0.02 <= thumb_tip.x <= gesture_c['thumb_tip'][0] + 0.02 and
                        gesture_c['index_tip'][0] - 0.02 <= index_tip.x <= gesture_c['index_tip'][0] + 0.02 and
                        gesture_c['middle_tip'][0] - 0.02 <= middle_tip.x <= gesture_c['middle_tip'][0] + 0.02 and
                        gesture_c['ring_tip'][0] - 0.02 <= ring_tip.x <= gesture_c['ring_tip'][0] + 0.02 and
                        gesture_c['pinky_tip'][0] - 0.02 <= pinky_tip.x <= gesture_c['pinky_tip'][0] + 0.02 and
                        gesture_c['thumb_tip'][1] - 0.02 <= thumb_tip.y <= gesture_c['thumb_tip'][1] + 0.02 and
                        gesture_c['index_tip'][1] - 0.02 <= index_tip.y <= gesture_c['index_tip'][1] + 0.02 and
                        gesture_c['middle_tip'][1] - 0.02 <= middle_tip.y <= gesture_c['middle_tip'][1] + 0.02 and
                        gesture_c['ring_tip'][1] - 0.02 <= ring_tip.y <= gesture_c['ring_tip'][1] + 0.02 and
                        gesture_c['pinky_tip'][1] - 0.02 <= pinky_tip.y <= gesture_c['pinky_tip'][1] + 0.02
                    ):
                        recognize_gesture('C')

                    # Verifica o gesto D
                    elif (
                        gesture_d['thumb_tip'][0] - 0.02 <= thumb_tip.x <= gesture_d['thumb_tip'][0] + 0.02 and
                        gesture_d['index_tip'][0] - 0.02 <= index_tip.x <= gesture_d['index_tip'][0] + 0.02 and
                        gesture_d['middle_tip'][0] - 0.02 <= middle_tip.x <= gesture_d['middle_tip'][0] + 0.02 and
                        gesture_d['ring_tip'][0] - 0.02 <= ring_tip.x <= gesture_d['ring_tip'][0] + 0.02 and
                        gesture_d['pinky_tip'][0] - 0.02 <= pinky_tip.x <= gesture_d['pinky_tip'][0] + 0.02 and
                        gesture_d['thumb_tip'][1] - 0.02 <= thumb_tip.y <= gesture_d['thumb_tip'][1] + 0.02 and
                        gesture_d['index_tip'][1] - 0.02 <= index_tip.y <= gesture_d['index_tip'][1] + 0.02 and
                        gesture_d['middle_tip'][1] - 0.02 <= middle_tip.y <= gesture_d['middle_tip'][1] + 0.02 and
                        gesture_d['ring_tip'][1] - 0.02 <= ring_tip.y <= gesture_d['ring_tip'][1] + 0.02 and
                        gesture_d['pinky_tip'][1] - 0.02 <= pinky_tip.y <= gesture_d['pinky_tip'][1] + 0.02
                    ):
                        recognize_gesture('D')

        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    end_time = time.time()  # Tempo final
    total_time = end_time - start_time
    print(f"Tempo total: {total_time} segundos")

cap.release()
cv2.destroyAllWindows()
