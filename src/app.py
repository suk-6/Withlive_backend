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
import logging
import os

# YOLO 모델과 가중치 로드
model = torch.hub.load('.', 'custom', path='./models/230218.pt', source='local', force_reload=True)

# 웹 소켓 서버 정보
HOST = '0.0.0.0'  # 호스트 주소
PORT = 20000  # 포트 번호

# LOGGER
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger()

# 바운딩 박스 설정
color = (0, 255, 0)
thickness = 2

# 바운딩 박스 이미지 저장
imageFolder = "./save-images"

if not os.path.exists(imageFolder):
    os.makedirs(imageFolder)

# 웹 소켓 서버 핸들러
async def serverHandler(websocket, path):
    # 이미지 카운터
    imageCounter = len(os.listdir(imageFolder))
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

            relativeXmin = xmin / imageWidth
            relativeYmin = ymin / imageHeight
            relativeWidth = (xmax - xmin) / imageWidth
            relativeHeight = (ymax - ymin) / imageHeight

            bboxCoords = {
                'requestTime': now.strftime('%Y-%m-%d_%H:%M:%S'),
                'left': relativeXmin,
                'top': relativeYmin,
                'width': relativeWidth,
                'height': relativeHeight,
                'conf': conf, 
                "label": int(label), 
                'position': -1, 
                'ratio': -1, 
                'imageWidth': imageWidth, 
                'imageHeight': imageHeight
            }

            xCenter = (xmin + xmax) / 2

            if xCenter < imageWidth / 2: # 왼쪽 아래
                bboxCoords['position'] = 0
            elif xCenter > imageWidth / 2: # 오른쪽 아래
                bboxCoords['position'] = 1
            elif xCenter == imageWidth / 2: # 정중dkd
                bboxCoords['position'] = 2
            else:
                bboxCoords['position'] = -1


            # 바운딩 박스의 크기 비율 계산
            bboxWidth = xmax - xmin
            bboxHeight = ymax - ymin
            bboxArea = bboxWidth * bboxHeight
            bboxRatio = (bboxArea / imageArea) * 100  # 퍼센트로 표현

            bboxCoords['ratio'] = bboxRatio

            annos.append(bboxCoords)

            cv2.rectangle(frameNp, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

        sendData = json.dumps(annos, default=str)

        LOGGER.info(sendData)

        # 객체인식 정보 전송
        await websocket.send(sendData)

        # 이미지 저장
        timeStamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        imageFilename = os.path.join(imageFolder, f'image_{timeStamp}.jpg')
        cv2.imwrite(imageFilename, frameNp, [cv2.IMWRITE_JPEG_QUALITY, 40])
        imageCounter += 1

        # 이미지 개수가 100개를 초과하면 가장 오래된 이미지 삭제
        if imageCounter > 100:
            # 저장된 이미지 파일 목록 가져오기
            imageFiles = sorted(os.listdir(imageFolder))
            
            # 가장 오래된 이미지 삭제
            oldestImage = os.path.join(imageFolder, imageFiles[0])
            os.remove(oldestImage)

            imageCounter -= 1

# Websocket Server
server = websockets.serve(serverHandler, HOST, PORT)
print(f'Host: {HOST}, Port: {PORT}')

# Loop
asyncio.get_event_loop().run_until_complete(server)
asyncio.get_event_loop().run_forever()
