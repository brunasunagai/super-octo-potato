import cv2 
import mediapipe as mp
import time

frame = cv2.VideoCapture(0)   # 0 é o num da webcam utilizada
handsDect = mp.solutions.hands
hands = handsDect.Hands()     #confidences=0.5, max num of hands=2, static mode=false são parametros padrão
connections = mp.solutions.drawing_utils

prevTime = 0    # tempo anterior
currTime = 0    # tempo atual

while True:
    success, image = frame.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    #print(results.multi_hand_landmarks)    #testa se está detectando a mão

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:   # para saber se é uma mão ou várias
            for id, points in enumerate(handLandmarks.landmark):  # id identifica o landmark correspondente
                #print(id,points)    #id é dado em núm decimal pq é tipo a razão do frame em que se encontra o pixel
                h, w, p = image.shape     # pega as dimensões do video 
                cx, cy = int(points.x*w), int(points.y*h)   # lm.x*w = pega a posição do pixel em x
                print(id,cx,cy)     # identifica qual a posição de cada landmark
                if id == 0:  # testa se é o ponto mais embaixo
                    cv2.circle(image, (cx,cy), 8, (255,0,100), cv2.FILLED)   # desenha um circulo no ponto 0 identificado
            connections.draw_landmarks(image, handLandmarks, handsDect.HAND_CONNECTIONS)     # aponta os landmarks das mãos, HAND_CONNECTIONS faz conexões

    currTime = time.time()     # pega o tempo atual
    fps = 1/(currTime-prevTime)   # cálculo do fps 
    prevTime = currTime
    cv2.putText(image,str(int(fps)), (10,460), cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)

    cv2.imshow("Imagem em tempo real", image)
    cv2.waitKey(1)