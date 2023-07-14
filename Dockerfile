FROM python:3.10.6

ENV TZ Asia/Seoul
ENV PYTHONIOENCODING UTF-8
ENV LC_CTYPE C.UTF-8

LABEL maintainer="https://suk.kr"

WORKDIR /server

COPY requirements.txt .
COPY src/ .

RUN apt update && apt install libgl1-mesa-glx ffmpeg libsm6 libxext6 -y
RUN pip install -r requirements.txt
RUN git clone https://github.com/ultralytics/yolov5

EXPOSE 20000

CMD [ "python", "./app.py" ]