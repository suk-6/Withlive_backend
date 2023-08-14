import asyncio
import cv2
import torch
from torchvision import transforms
import websockets
import base64
from PIL import Image
import numpy as np
import json
from datetime import datetime

# YOLO 모델과 가중치 로드
model = torch.hub.load('./yolov5', 'custom', path='./models/230218.pt', source='local', force_reload=True)

# 웹 소켓 서버 정보
HOST = '0.0.0.0'  # 호스트 주소
PORT = 20000  # 포트 번호

# 웹 소켓 서버 핸들러
async def serverHandler(websocket, path):
    while True:
        # 클라이언트로부터 프레임 수신
        frameBase64 = await websocket.recv()
        now = datetime.now()

        # 프레임 디코딩 및 처리
        frameData = base64.b64decode(frameBase64)
        frameNp = cv2.imdecode(np.frombuffer(frameData, np.uint8), cv2.IMREAD_COLOR)
        framePil = Image.fromarray(frameNp)

        # 객체 탐지 수행
        results = model(framePil)

        # 객체인식 정보 추출
        annos = []

        imageWidth = frameNp.shape[1]
        imageHeight = frameNp.shape[0]
        imageArea = imageWidth * imageHeight

        for bbox in zip(results.xyxy[0]):
            xmin, ymin, xmax, ymax, conf, label = bbox[0].tolist()
            bboxCoords = {'requestTime': now.strftime('%Y-%m-%d_%H:%M:%S'), 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'conf': conf, "label": int(label)}

            xCenter = (xmin + xmax) / 2
            yCenter = (ymin + ymax) / 2

            # 바운딩 박스 위치 계산
            if xCenter < imageWidth / 2 and yCenter < imageHeight / 2: # 왼쪽 위
                bboxCoords['position'] = 0
            elif xCenter > imageWidth / 2 and yCenter < imageHeight / 2: # 오른쪽 위
                bboxCoords['position'] = 1
            elif xCenter < imageWidth / 2 and yCenter > imageHeight / 2: # 왼쪽 아래
                bboxCoords['position'] = 2
            elif xCenter > imageWidth / 2 and yCenter > imageHeight / 2: # 오른쪽 아래
                bboxCoords['position'] = 3
            else:
                bboxCoords['position'] = -1


            # 바운딩 박스의 크기 비율 계산
            bboxWidth = xmax - xmin
            bboxHeight = ymax - ymin
            bboxArea = bboxWidth * bboxHeight
            bboxRatio = (bboxArea / imageArea) * 100  # 퍼센트로 표현

            bboxCoords['ratio'] = bboxRatio

            annos.append(bboxCoords)

        sendData = json.dumps(annos, default=str)

        print(sendData)

        # 객체인식 정보 전송
        await websocket.send(sendData)

# Websocket Server
server = websockets.serve(serverHandler, HOST, PORT)
print(f'Host: {HOST}, Port: {PORT}')

# Loop
asyncio.get_event_loop().run_until_complete(server)
asyncio.get_event_loop().run_forever()
