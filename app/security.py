"""Security utilities for the Transient Recommender API."""

from fastapi import Depends, HTTPException, status, Cookie, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import secrets
import string
import logging

from .database import get_db
from . import models

# Initialize logger
logger = logging.getLogger(__name__)

# Token data model
class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None

# Security constants
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

# Security utils
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    """Verify password against hashed version."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password for storing."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_api_key():
    """Generate a random API key."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(32))

async def get_token_from_cookie(access_token: Optional[str] = Cookie(None)):
    """Get access token from cookie."""
    if not access_token:
        return None
    if access_token.startswith("Bearer "):
        return access_token[7:]
    return access_token

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Get current user from request."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = None
    
    # First try to get token from Authorization header
    authorization: str = request.headers.get("Authorization")
    if authorization:
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                token = None
        except ValueError:
            token = None
    
    # If no header token, try to get from cookie
    if not token:
        token = request.cookies.get("access_token")
        if token and token.startswith("Bearer "):
            token = token[7:]  # Remove "Bearer " prefix
        elif token and not token.startswith("Bearer"):
            # Token might be without Bearer prefix in cookie
            pass
    
    if not token:
        logger.debug("No token found in request")
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if username is None or user_id is None:
            logger.debug(f"Invalid token payload: username={username}, user_id={user_id}")
            raise credentials_exception
            
        token_data = TokenData(username=username, user_id=user_id)
    except JWTError as e:
        logger.debug(f"JWT decode error: {e}")
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.id == token_data.user_id).first()
    if user is None:
        logger.debug(f"User not found for user_id: {token_data.user_id}")
        raise credentials_exception
        
    logger.debug(f"Successfully authenticated user: {user.username}")
    return user

async def get_current_user_optional(request: Request, db: Session = Depends(get_db)):
    """Get current user from request - return None if not authenticated."""
    try:
        return await get_current_user(request, db)
    except HTTPException:
        return None

def verify_data_sharing_consent(user: models.User):
    """Verify that the user has consented to data sharing."""
    if not user.data_sharing_consent:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has not consented to data sharing"
        )

async def get_current_user_from_api_key(api_key: str, db: Session = Depends(get_db)):
    """Get current user from API key."""
    user = db.query(models.User).filter(models.User.api_key == api_key).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user 