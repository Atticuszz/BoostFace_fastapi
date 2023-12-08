# coding=utf-8
from fastapi import APIRouter
from gotrue import AuthResponse, Session

from fastapi_app.client import supabase_client
from fastapi_app.model import UserLogin

auth_router = APIRouter(prefix="/auth", tags=["auth"])


@auth_router.post("/login")
async def login(user: UserLogin) -> Session:
    response: AuthResponse = await supabase_client.sign_in(
        email=user.email, password=user.password
    )
    return response.session
