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

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

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
                    # Atraso de 2 segundos
                    if time.time() - start_time > 2:
                        # Exibe a posição relativa dos dedos com duas casas decimais
                        print('Posição do polegar: {:.2f}, {:.2f}'.format(thumb_tip.x, thumb_tip.y))
                        print('Posição do indicador: {:.2f}, {:.2f}'.format(index_tip.x, index_tip.y))
                        print('Posição do dedo médio: {:.2f}, {:.2f}'.format(middle_tip.x, middle_tip.y))
                        print('Posição do dedo anelar: {:.2f}, {:.2f}'.format(ring_tip.x, ring_tip.y))
                        print('Posição do dedo mindinho: {:.2f}, {:.2f}'.format(pinky_tip.x, pinky_tip.y))
                        print(' ')
                        # Reinicia o tempo de início
                        start_time = time.time()

        cv2.imshow('Detecção de Mão', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
