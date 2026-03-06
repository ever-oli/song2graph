FROM python:3.10-bullseye

RUN apt update
RUN apt install -y rubberband-cli make automake gcc g++ python3-dev gfortran build-essential wget libsndfile1 ffmpeg

RUN pip install --upgrade pip

COPY . /song2graph
WORKDIR /song2graph

RUN pip install -r requirements.txt

RUN mkdir -p input processed separated library
