# BoostFace_fastapi

## Introduction

- cloud compute for extract and search face embedding
- db backend for desktop and mobile app

## deployment

- docker compose
  - fastapi cloud compute
  - milvus-standalone

- cloud service
  - digital ocean
    - vps

## architecture

- fastapi container
  - main process
    - identify-worker sub process
      - extract
        - arcface onnx
      - register or search
        - milvus
    - fastapi thread
      - basic request
      - get face_image from front-end and add to worker queue be shared with identify-worker
- milvus container
  - milvus-standalone
    - milvus
    - minio
    - milvus-etcd

## Project Process🌈

1. locally test fastapi cloud compute docker compose
  1. login










