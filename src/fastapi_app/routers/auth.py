# coding=utf-8
from fastapi import APIRouter, HTTPException
from gotrue import AuthResponse, Session
from gotrue.errors import AuthApiError

from fastapi_app.client import supabase_client
from fastapi_app.model import UserLogin

auth_router = APIRouter(prefix="/auth", tags=["auth"])


@auth_router.post("/login")
async def login(user: UserLogin) -> Session:
    """sign in with email and password"""
    try:
        response: AuthResponse = await supabase_client.sign_in(user.email, user.password)
    except AuthApiError:
        raise HTTPException(
            status_code=400,
            detail="Invalid email or password")
    return response.session


@auth_router.post("/refresh-token")
async def refresh_token(refresh_token: str) -> Session:
    """refresh token"""
    try:
        new_session: AuthResponse = await supabase_client.refresh_session(refresh_token)
        return new_session.session
    except AuthApiError:
        raise HTTPException(status_code=401, detail="Could not refresh token")
