from pydantic import BaseModel


# coding=utf-8


class UserLogin(BaseModel):
    email: str
    password: str


class identifyResult(BaseModel):
    id: str
    name: str
    time: str
