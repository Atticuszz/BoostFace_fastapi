# coding=utf-8
import asyncio
import os

from dotenv import load_dotenv
from gotrue import AuthResponse
from postgrest import APIResponse
from starlette.websockets import WebSocket
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from fastapi_app.dependence import retry
from fastapi_app.internal import logger
from supabase_py_async import AsyncClient, create_client
from supabase_py_async.lib.client_options import ClientOptions


class SupaBase:
    def __init__(self):
        self.client: AsyncClient | None = None

    async def set_session(self, access_token: str, refresh_token: str) -> AuthResponse:
        """
        set session
        :param access_token:
        :param refresh_token:
        :return:AuthResponse
        :exception AuthSessionMissingError: Could not set session
        """
        response: AuthResponse = await self.client.auth.set_session(access_token=access_token,
                                                                    refresh_token=refresh_token)
        return response

    async def refresh_session(self, refresh_token: str) -> AuthResponse:
        """
        refresh token
        :exception AuthApiError: Could not refresh token
        """
        new_session: AuthResponse = await self.client.auth.refresh_session(refresh_token)

        return new_session

    async def sign_in(self, email: str, password: str) -> AuthResponse:
        """
        sign in with email and password
        :exception AuthApiError: Invalid email or password
        """
        response: AuthResponse = await self.client.auth.sign_in_with_password(
            {'email': email, 'password': password})

        return response

    @retry
    async def get_table(self, name: str, columns: list[str] | None = None) -> list[dict]:
        if columns is None:
            columns = ["*"]
        select_params: str = ",".join(columns)
        response: APIResponse = await self.client.table(name).select(select_params).execute()
        return response.data

    @retry
    async def upsert(self, name: str, data: dict) -> list[dict]:
        response: APIResponse = await self.client.table(name).upsert(data).execute()
        return response.data

    @retry
    async def delete(self, name: str, uuid: str) -> list[dict]:
        response: APIResponse = await self.client.table(
            name).delete().eq("uuid", uuid).execute()
        return response.data

    async def multi_requests(self, name: str, data: list[dict], options: str, chunk_size: int = 4000):

        for i in range(0, len(data), chunk_size):
            chunk_data = data[i:i + chunk_size]
            match options:
                case "upsert":
                    tasks = [asyncio.ensure_future(self.upsert(name, item))
                             for item in chunk_data]
                case 'delete':
                    tasks = [
                        asyncio.ensure_future(
                            self.delete(name,
                                        item["uuid"])) for item in chunk_data]
                case _:
                    raise ValueError("options must be 'upsert' or 'delete'")
            await asyncio.gather(*tasks)
        return


class WebSocketManager:
    """ WebSocket manager"""

    def __init__(self):
        self.active_connections: dict[str, tuple[WebSocket, str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str, category: str):
        """Connect with category."""
        await websocket.accept()
        self.active_connections[client_id] = (websocket, category)

    async def disconnect(self, client_id: str):
        """Disconnect."""
        await self.active_connections[client_id][0].close()
        del self.active_connections[client_id]

    async def broadcast(self, message: str, category: str = None):
        """Broadcast message to all connections or specific category."""
        try:
            for websocket, cat in self.active_connections.values():
                if category is None or cat == category:
                    await websocket.send_text(message)
        except (ConnectionClosedOK, ConnectionClosedError):
            logger.warn("broadcast to the websocket is closed ")


web_socket_manager = WebSocketManager()
supabase_client = SupaBase()


async def main():
    load_dotenv()
    url: str = os.getenv("SUPABASE_URL")
    key: str = os.getenv("SUPABASE_KEY")
    supabase_client.client = await create_client(url, key, options=ClientOptions(
        postgrest_client_timeout=10, storage_client_timeout=10))
    data = await supabase_client.get_table("task_done_list")
    print(data)


if __name__ == "__main__":
    asyncio.run(main())
    # 将数据转换为json数据
