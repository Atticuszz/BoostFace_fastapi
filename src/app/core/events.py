# coding=utf-8
"""
life span events
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from supabase_py_async import create_client
from supabase_py_async.lib.client_options import ClientOptions
from .config import logger
from .supabase_client import supabase_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ life span events"""
    load_dotenv()
    url: str = os.getenv("SUPABASE_URL")
    key: str = os.getenv("SUPABASE_KEY")
    supabase_client.client = await create_client(url, key, options=ClientOptions(
        postgrest_client_timeout=10, storage_client_timeout=10))
    assert supabase_client.client is not None
    logger.info("supabase client created successfully")
    yield
