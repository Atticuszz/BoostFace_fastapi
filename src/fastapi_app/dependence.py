# Retry decorator
from asyncio import sleep
from functools import wraps

from fastapi import Header, HTTPException


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


async def validate_user(authorization: str | None = Header(None)):
    """
    refresh session
    :param authorization:
    """
    from fastapi_app.client import supabase_client
    if not authorization:
        raise HTTPException(status_code=401, detail="Unauthorized")
    tokens = authorization.split(" ")

    if len(tokens) != 2:
        raise HTTPException(status_code=401,
                            detail="Invalid authorization header format")

    access_token, refresh_token = tokens
    try:
        session = await supabase_client.set_session(access_token=access_token, refresh_token=refresh_token)
        return session
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
