from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    full_name: Optional[str] = None
    email: str
    password: str
    role: str = "Speaker"


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    full_name: Optional[str]
    email: EmailStr
    role: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenRefresh(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
