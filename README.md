---
title: Mobilenet Api
emoji: 👀
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
---

# MobileNetV3 Classification API

## Install
pip install -r requirements.txt

## Benchmark
python scripts/benchmark.py

## Export ONNX
python scripts/export_onnx.py

## Quantize
python scripts/quantize.py

## Run API
uvicorn app.main:app --reload

## Test
python -m pytest

## Docker
docker build -t mobilenet-api .

docker run -p 7860:7860 mobilenet-api

## Predict
curl -X POST http://127.0.0.1:8000/predict -F "file=@test.jpg"