# Retry decorator
from asyncio import sleep
from functools import wraps

from fastapi import Header, HTTPException
from gotrue import AuthResponse
from gotrue.errors import AuthSessionMissingError


def retry(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        for i in range(3):  # Retry up to 3 times
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print("*args:", *args, "**kwargs:", **kwargs)
                print(f"\nAn error occurred: {e}. Retrying...")
                await sleep(0.01)  # Wait for 0.01 seconds before retrying
        print("Failed after 3 retries.")

    return wrapper


async def validate_user(authorization: str | None = Header(None), refresh_token: str | None = Header(None)):
    """
    Validate user session with access and refresh tokens
    :param authorization: Authorization header containing the access token
    :param refresh_token: Header containing the refresh token
    """
    from fastapi_app.internal.client import supabase_client
    if not authorization or not refresh_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    tokens = authorization.split(" ")
    if len(tokens) != 2 or tokens[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    access_token = tokens[1]

    try:
        # Assuming set_session requires both access and refresh tokens
        response: AuthResponse = await supabase_client.set_session(access_token=access_token,
                                                                   refresh_token=refresh_token)
        return response.session
    except AuthSessionMissingError as e:
        raise HTTPException(status_code=401, detail=e.message)
