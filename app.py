import asyncio
import cv2
import torch
from torchvision import transforms
import websockets
import base64
from PIL import Image
import numpy as np
import json


# YOLO 모델과 가중치 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 웹 소켓 서버 정보
HOST = '0.0.0.0'  # 호스트 주소
PORT = 20000  # 포트 번호

# 웹 소켓 서버 핸들러
async def server_handler(websocket, path):
    while True:
        # 클라이언트로부터 프레임 수신
        frameBase64 = await websocket.recv()

        # 프레임 디코딩 및 처리
        frameData = base64.b64decode(frameBase64)
        frameNp = cv2.imdecode(np.frombuffer(frameData, np.uint8), cv2.IMREAD_COLOR)
        framePil = Image.fromarray(frameNp)

        # 객체 탐지 수행
        results = model(framePil)

        # 객체인식 정보 추출
        annos = []

        for bbox in zip(results.xyxy[0]):
            xmin, ymin, xmax, ymax, conf, label = bbox[0].tolist()
            bboxCoords = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'conf': conf, "label": int(label)}
            annos.append(bboxCoords)

        sendData = json.dumps(annos)

        print(sendData)

        # 객체인식 정보 전송
        await websocket.send(sendData)

# Websocket Server
server = websockets.serve(server_handler, HOST, PORT)
print(f'Host: {HOST}, Port: {PORT}')

# Loop
asyncio.get_event_loop().run_until_complete(server)
asyncio.get_event_loop().run_forever()
