from pydantic import BaseModel

# coding=utf-8
class UserLogin(BaseModel):
    email: str
    password: str
