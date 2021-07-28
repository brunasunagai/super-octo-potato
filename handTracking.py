import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)   # 0 é o num da webcam utilizada

mpHands = mp.solutions.hands
hands = mpHands.Hands()     #confidences=0.5, max num of hands=2, static mode=false
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)    #testa se está detectando a mão

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:   # para saber se é uma mão ou várias
            for id, lm in enumerate(handLms.landmark):  # id identifica o landmark correspondente
                #print(id,lm)    #id é dado em núm decimal pq é tipo a razão do frame em que se encontra o pixel
                h, w, c = img.shape     # pega as dimensões do video
                cx, cy = int(lm.x*w), int(lm.y*h)   # lm.x*w = pega a posição do pixel em x
                print(id,cx,cy)     # identifica qual a posição de cada landmark
                if id ==0:  # testa se é o ponto mais embaixo
                    cv2.circle(img, (cx,cy), 15, (255,0,100), cv2.FILLED)   # desenha um circulo no ponto 0 identificado
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)     # aponta os landmarks das mãos, HAND_CONNECTIONS faz conexões

    cTime = time.time()     # pega o tempo atual
    fps = 1/(cTime-pTime)   # cálculo do fps 
    pTime = cTime
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,100),3)

    cv2.imshow("Imagem da webcam", img)
    cv2.waitKey(1)