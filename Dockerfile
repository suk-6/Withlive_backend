FROM ultralytics/yolov5:latest

ENV TZ Asia/Seoul
ENV PYTHONIOENCODING UTF-8
ENV LC_CTYPE C.UTF-8

LABEL maintainer="https://suk.kr"

COPY requirements.txt ../
COPY src/app.py .
COPY src/models/ ./models/

RUN pip install websockets

EXPOSE 20000

CMD [ "python3", "./app.py" ]
