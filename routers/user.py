from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models.user import User
from utils.db import get_db

router = APIRouter(prefix="/api/user", tags=["user"])


class CreateUserRequest(BaseModel):
    name: str


class UserResponse(BaseModel):
    id: int
    name: str
    created_at: object | None = None


@router.post("", response_model=UserResponse)
def create_user(req: CreateUserRequest, db: Session = Depends(get_db)):
    user = User(name=req.name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse(id=user.id, name=user.name, created_at=user.created_at)


@router.get("", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)):
    users = User.list_all(db)
    return [UserResponse(id=u.id, name=u.name, created_at=u.created_at) for u in users]
