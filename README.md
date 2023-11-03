# BoostFace

## apps

### mobile ->user

- frontend: react native
- backend: fastapi
  - basic functions
    - login
    - register to db
    - state notification
    - record check
  - deeplearning
    - face register
      - detect
      - extract and register to vector db
      - check
    - face recognition
      - detect
      - identify

### web/desktop ->admin

- frontend: vue
- backend: fastapi
  - basic functions
    - login
    - show total state from db
  - deeplearning
    - process video stream and show results
    - face recognition
      - detect
      - identify

### modules

- db
  - vector db
    - milvus_client -> self-host docker🐟
  - sql db
    - supabase -> cloud service☁️

- deeplearning
  - boostface-A for attendance
    - identify
      - extract
        - arcface onnx
      - match by milvus
    - detect
      - scrfd onnx

  - boostface-R for register
    - detect
      - scrfd onnx
    - extract
      - arcface onnx
    - register
      - milvus


- mobile-fastapi- basic fun

- mobile-fastapi
  - register

- web-fastapi- all in one
  - attendance
