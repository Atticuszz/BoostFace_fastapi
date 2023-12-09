# coding=utf-8
from fastapi import APIRouter, HTTPException
from gotrue import AuthResponse, Session
from gotrue.errors import AuthApiError

from fastapi_app.internal import logger
from fastapi_app.internal.client import supabase_client
from fastapi_app.internal.model import UserLogin

auth_router = APIRouter(prefix="/auth", tags=["auth"])


@auth_router.post("/login")
async def login(user: UserLogin) -> Session:
    """sign in with email and password"""
    try:
        response: AuthResponse = await supabase_client.sign_in(user.email, user.password)
        logger.info(f"User {user.email} logged in")
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
        logger.info(f"User {new_session.user.email} refreshed token")
        return new_session.session
    except AuthApiError:
        raise HTTPException(status_code=401, detail="Could not refresh token")
