import asyncio
import websockets
import base64
import json

# 웹 소켓 서버 정보
SERVER_URI = 'ws://localhost:20000'

# 웹 소켓 클라이언트
async def client():
    async with websockets.connect(SERVER_URI) as websocket:
        with open('./images/0125_frame_000032.jpg', 'rb') as img:
            base64String = base64.b64encode(img.read())
            await websocket.send(base64String)

        data = await websocket.recv()
        jsonData = json.loads(data)

        print(jsonData)

        await websocket.close()

asyncio.get_event_loop().run_until_complete(client())