import os
import subprocess
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase_py_async import create_client
from supabase_py_async.lib.client_options import ClientOptions

from fastapi_app.client import supabase_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    url: str = os.getenv("SUPABASE_URL")
    key: str = os.getenv("SUPABASE_KEY")
    supabase_client.client = await create_client(url, key, options=ClientOptions(
        postgrest_client_timeout=10, storage_client_timeout=10))
    assert supabase_client.client is not None
    yield


def create_app() -> FastAPI:
    # 初始化 FastAPI 和 StrapiClient
    app = FastAPI(lifespan=lifespan)
    # 设置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the routers
    from fastapi_app.routers import auth_router
    app.include_router(auth_router)
    return app


app = create_app()


@app.post("/add_test")
async def add_test(data: dict):
    pass


def server_run(debug: bool = False, port: int = 5000):
    if not debug:
        # Run FastAPI with reload

        subprocess.Popen(["uvicorn", "app:app", "--host",
                          "127.0.0.1", "--port", str(port), "--reload"])
    else:

        uvicorn.run(app, port=port)


if __name__ == "__main__":
    server_run(True)
