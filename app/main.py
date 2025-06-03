"""Main FastAPI application."""

from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, Request, Form, File, UploadFile, Response, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import os
import logging
import re
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
import io # For pydub

import speech_recognition as sr # For speech-to-text
from pydub import AudioSegment # For audio conversion
import smtplib
import secrets

# Email imports - the correct class names are MIMEText and MIMEMultipart
try:
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
except ImportError:
    try:
        # Alternative import path for older Python versions
        from email.MIMEText import MIMEText
        from email.MIMEMultipart import MIMEMultipart
    except ImportError:
        # Final fallback - disable email functionality
        logger.warning("Email modules not available - password reset emails will not work")
        MIMEText = None
        MIMEMultipart = None

from . import models, security, recommender
from .database import engine, get_db
from .security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user,
    get_current_user_from_api_key,
    verify_data_sharing_consent,
    verify_password,
    get_password_hash,
    create_access_token,
    generate_api_key,
    get_current_user_optional,
)
from .models import User
from .recommender import WebRecommender, NoObjectsAvailableError
from .feature_extractor import extract_features_for_recent_objects, get_last_extraction_run, should_run_feature_extraction
from .pending_votes import get_pending_objects_for_science_case, remove_pending_vote
from .database import SessionLocal
from .anomaly_service import anomaly_service
from .filter_manager import filter_manager

# Configure logging
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# Basic configuration (can be expanded with handlers, formatters for production)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Main app for browser routes - configure for proxy
app = FastAPI(
    title="Transient Recommender API",
    # Configure for reverse proxy
    root_path="",
    servers=[
        {"url": "https://transientrecommender.org", "description": "Production server"},
        {"url": "http://localhost:8080", "description": "Development server"}
    ]
)

# API sub-app (no CSRF)
api_app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize recommender
feature_bank_path = os.getenv("FEATURE_BANK_PATH", "data/feature_bank.csv")
recommender_engine = recommender.WebRecommender(feature_bank_path)

# Security utilities
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Custom middleware to log all requests
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Request failed: {request.method} {request.url} - Error: {e}", exc_info=True)
            raise

# Apache proxy headers middleware for HTTPS support
class ApacheProxyHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Trust Apache proxy headers for HTTPS
        if "X-Forwarded-Proto" in request.headers:
            forwarded_proto = request.headers["X-Forwarded-Proto"]
            if forwarded_proto == "https":
                request.scope["scheme"] = "https"
                # Also update the server info to reflect HTTPS
                if "server" in request.scope:
                    server = list(request.scope["server"])
                    server[1] = 443  # HTTPS port
                    request.scope["server"] = tuple(server)
        
        # Also check for other common Apache headers
        if "X-Forwarded-SSL" in request.headers and request.headers["X-Forwarded-SSL"] == "on":
            request.scope["scheme"] = "https"
            if "server" in request.scope:
                server = list(request.scope["server"])
                server[1] = 443  # HTTPS port
                request.scope["server"] = tuple(server)
        
        # Handle X-Forwarded-Host
        if "X-Forwarded-Host" in request.headers:
            request.scope["headers"] = [
                (k, v) for k, v in request.scope["headers"] 
                if k != b"host"
            ] + [(b"host", request.headers["X-Forwarded-Host"].encode())]
            
        response = await call_next(request)
        return response

# Add the middleware to both apps
app.add_middleware(ApacheProxyHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
api_app.add_middleware(ApacheProxyHeadersMiddleware) 
api_app.add_middleware(RequestLoggingMiddleware)

# Move token route to the beginning to test if position matters
@app.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Get access token for login."""
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    """Handle Chrome DevTools request to prevent 404 errors."""
    return {"message": "No DevTools configuration"}



@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: models.User = Depends(get_current_user_optional)):
    """Home page."""
    return templates.TemplateResponse("index.html", {"request": request, "current_user": current_user})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render login page."""
    # Temporary dummy CSRF token for testing
    csrf_token = "dummy-csrf-token"
    return templates.TemplateResponse("login.html", {"request": request, "csrf_token": csrf_token})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Render registration page."""
    # Temporary dummy CSRF token for testing
    csrf_token = "dummy-csrf-token"
    return templates.TemplateResponse("register.html", {"request": request, "csrf_token": csrf_token})

@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    data_sharing_consent: bool = Form(False),
    science_interests: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Register a new user."""
    # Check if username or email already exists
    existing_user = db.query(models.User).filter(
        (models.User.username == username) | (models.User.email == email)
    ).first()
    
    if existing_user:
        csrf_token = "dummy-csrf-token"
        if existing_user.username == username:
            return templates.TemplateResponse(
                "register.html", 
                {
                    "request": request, 
                    "csrf_token": csrf_token,
                    "error": "Username already exists."
                }
            )
        else:
            return templates.TemplateResponse(
                "register.html", 
                {
                    "request": request, 
                    "csrf_token": csrf_token,
                    "error": "Email already exists."
                }
            )
    
    # Create new user
    hashed_password = get_password_hash(password)
    # Accept science_interests as a JSON array or string
    interests = None
    if science_interests:
        if isinstance(science_interests, str):
            try:
                interests = json.loads(science_interests)
            except Exception:
                interests = [science_interests]
        elif isinstance(science_interests, list):
            interests = science_interests
    new_user = models.User(
        username=username,
        email=email,
        password_hash=hashed_password,
        data_sharing_consent=data_sharing_consent,
        science_interests=interests
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username, "user_id": new_user.id},
        expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/recommendations", status_code=303)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Handle login form submission."""
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        csrf_token = "dummy-csrf-token"
        return templates.TemplateResponse(
            "login.html", 
            {
                "request": request, 
                "csrf_token": csrf_token,
                "error": "Invalid username or password."
            }
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/recommendations", status_code=303)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.get("/logout")
async def logout():
    """Log out the user."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="access_token")
    return response

def send_password_reset_email(email: str, reset_token: str, base_url: str = "http://localhost:8080"):
    """Send password reset email."""
    try:
        # Check if email modules are available
        if MIMEText is None or MIMEMultipart is None:
            logger.warning("Email modules not available - cannot send password reset email")
            return False
        
        # Email configuration - in production, use environment variables
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        sender_email = os.getenv("SENDER_EMAIL", "noreply@transientrecommender.org")
        sender_password = os.getenv("SENDER_PASSWORD", "")
        
        if not sender_password:
            logger.warning("No SENDER_PASSWORD configured, skipping email send")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = "Password Reset - Transient Recommender"
        
        reset_url = f"{base_url}/reset-password?token={reset_token}"
        
        body = f"""
        You have requested a password reset for your Transient Recommender account.
        
        Click the link below to reset your password:
        {reset_url}
        
        This link will expire in 1 hour.
        
        If you did not request this password reset, please ignore this email.
        
        Best regards,
        The Transient Recommender Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, email, text)
        server.quit()
        
        logger.info(f"Password reset email sent to {email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send password reset email: {e}")
        return False

@api_app.post("/auth/request-password-reset")
async def request_password_reset(
    request: Request,
    db: Session = Depends(get_db)
):
    """Request a password reset email."""
    data = await request.json()
    email = data.get("email")
    
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
    user = db.query(models.User).filter(models.User.email == email).first()
    
    # Always return success to prevent email enumeration
    # but only send email if user exists
    if user:
        # Generate secure random token
        reset_token = secrets.token_urlsafe(32)
        
        # Create password reset token record
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        # Invalidate any existing tokens for this user
        db.query(models.PasswordResetToken).filter(
            models.PasswordResetToken.user_id == user.id,
            models.PasswordResetToken.used == False
        ).update({"used": True, "used_at": datetime.utcnow()})
        
        reset_token_record = models.PasswordResetToken(
            user_id=user.id,
            token=reset_token,
            expires_at=expires_at
        )
        db.add(reset_token_record)
        db.commit()
        
        # Send email
        base_url = str(request.base_url).rstrip('/')
        send_password_reset_email(email, reset_token, base_url)
    
    return {"message": "If an account with that email exists, a password reset link has been sent."}

@app.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str, db: Session = Depends(get_db)):
    """Render password reset page."""
    # Verify token
    reset_token = db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.token == token,
        models.PasswordResetToken.used == False,
        models.PasswordResetToken.expires_at > datetime.utcnow()
    ).first()
    
    if not reset_token:
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "error": "Invalid or expired reset token.",
            "token": None
        })
    
    return templates.TemplateResponse("reset_password.html", {
        "request": request,
        "token": token,
        "error": None
    })

@app.post("/reset-password")
async def reset_password(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Handle password reset form submission."""
    if password != confirm_password:
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "token": token,
            "error": "Passwords do not match."
        })
    
    if len(password) < 6:
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "token": token,
            "error": "Password must be at least 6 characters long."
        })
    
    # Find and validate token
    reset_token = db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.token == token,
        models.PasswordResetToken.used == False,
        models.PasswordResetToken.expires_at > datetime.utcnow()
    ).first()
    
    if not reset_token:
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "token": token,
            "error": "Invalid or expired reset token."
        })
    
    # Update user password
    user = reset_token.user
    user.password_hash = get_password_hash(password)
    
    # Mark token as used
    reset_token.used = True
    reset_token.used_at = datetime.utcnow()
    
    db.commit()
    
    return templates.TemplateResponse("reset_password.html", {
        "request": request,
        "token": None,
        "success": "Password successfully reset! You can now log in with your new password."
    })

@app.get("/targets", response_class=HTMLResponse)
async def targets_page(
    request: Request,
    current_user: models.User = Depends(get_current_user)
):
    """Render the targets page."""
    logger.info(f"Targets page accessed by authenticated user: {current_user.username} (ID: {current_user.id})")
    
    context = {"request": request, "username": current_user.username, "current_user": current_user}
    logger.info(f"Template context for targets page: {context.keys()}")
    logger.info(f"current_user in context: {context.get('current_user') is not None}")
    
    return templates.TemplateResponse("targets.html", context)

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    current_user: models.User = Depends(get_current_user)
):
    """Profile page."""
    return templates.TemplateResponse("profile.html", {"request": request, "current_user": current_user})

@app.get("/preferences", response_class=HTMLResponse)
async def preferences_page(
    request: Request,
    current_user: models.User = Depends(get_current_user)
):
    """Preferences page."""
    return templates.TemplateResponse("preferences.html", {"request": request, "current_user": current_user})

@app.get("/history", response_class=HTMLResponse)
async def history_page(
    request: Request,
    current_user: models.User = Depends(get_current_user)
):
    """History page."""
    return templates.TemplateResponse("history.html", {"request": request, "current_user": current_user})

@app.get("/recommendations", response_class=HTMLResponse)
async def recommendations_page(
    request: Request,
    ztfid: Optional[str] = None,
    current_user: models.User = Depends(get_current_user)
):
    """Recommendations page."""
    context = {"request": request, "current_user": current_user}
    if ztfid:
        context["start_ztfid"] = ztfid
    return templates.TemplateResponse("recommendations.html", context)

@app.get("/algorithms", response_class=HTMLResponse)
async def algorithms_page(
    request: Request,
    current_user: models.User = Depends(get_current_user)
):
    """Algorithm management page (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required to access algorithm management",
        )
    
    return templates.TemplateResponse("algorithms.html", {"request": request, "current_user": current_user})

@api_app.get("/recommendations")
async def get_recommendations(
    science_case: str = "snia-like",
    count: int = 10,
    obs_telescope: Optional[str] = None,
    obs_days: Optional[int] = None,
    obs_mag_limit: Optional[float] = None,
    start_ztfid: Optional[str] = None,
    realtime_mode: bool = False,
    recent_days: int = 7,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get recommendations for the current user."""
    logger.info(f"API: get_recommendations called with start_ztfid={start_ztfid}, science_case={science_case}, realtime_mode={realtime_mode}")
    if realtime_mode:
        logger.info(f"Real-time mode: filtering to objects with detections in last {recent_days} days")
    
    try:
        recommendations = recommender_engine.get_recommendations(
            db, current_user.id, count, science_case,
            obs_telescope, obs_days, obs_mag_limit,
            start_ztfid=start_ztfid, realtime_mode=realtime_mode, recent_days=recent_days
        )
        logger.info(f"API: returning {len(recommendations)} recommendations")
        return recommendations
    
    except NoObjectsAvailableError as e:
        logger.warning(f"No objects available for user {current_user.id}: {e}")
        
        # Return a structured error response with constraint details
        error_response = {
            "error": "no_objects_available",
            "message": str(e),
            "active_constraints": e.active_constraints,
            "suggestions": []
        }
        
        # Add helpful suggestions based on active constraints
        if e.active_constraints.get('telescope'):
            error_response["suggestions"].append("Try selecting a different telescope or removing the telescope constraint")
        
        if e.active_constraints.get('magnitude_limit'):
            error_response["suggestions"].append(f"Try increasing the magnitude limit (currently {e.active_constraints['magnitude_limit']})")
        
        if e.active_constraints.get('observation_days') is not None:
            if e.active_constraints['observation_days'] == 0:
                error_response["suggestions"].append("Try selecting a different observation date or removing the 'tonight' constraint")
            else:
                error_response["suggestions"].append(f"Try changing the observation date (currently {e.active_constraints['observation_days']} days from now)")
        
        if e.active_constraints.get('excluded_voted_objects', 0) > 0:
            error_response["suggestions"].append("You've voted on many objects! You can view them in your History page")
        
        # Return a 422 (Unprocessable Entity) status code to indicate valid request but no results
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=error_response)

@api_app.post("/vote")
async def vote(
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Handle user votes - supports toggle behavior and multiple vote types per object."""
    data = await request.json()
    ztfid = data.get("ztfid")
    vote_type = data.get("vote")
    science_case = data.get("science_case", "snia-like")
    vote_payload_details = data.get("metadata", {})
    
    if not ztfid or not vote_type:
        raise HTTPException(status_code=400, detail="Missing ztfid or vote_type")
    
    # Check if this specific vote type already exists
    existing_vote = db.query(models.Vote).filter(
        models.Vote.user_id == current_user.id,
        models.Vote.ztfid == ztfid,
        models.Vote.vote_type == vote_type
    ).first()
    
    if existing_vote:
        # TOGGLE BEHAVIOR: If vote already exists, remove it (unlike/untarget)
        db.delete(existing_vote)
        logger.info(f"Removed {vote_type} vote for {ztfid} by user {current_user.id} (toggle off)")
        vote_action = "removed"
    else:
        # Create new vote of this type
        new_vote = models.Vote(
            user_id=current_user.id,
            ztfid=ztfid,
            vote_type=vote_type,
            science_case=science_case,
            vote_details=vote_payload_details
        )
        db.add(new_vote)
        logger.info(f"Created new {vote_type} vote for {ztfid} by user {current_user.id}")
        vote_action = "added"
        
        # Remove any pending votes for this object from the current science case
        # This ensures that when a user votes on a pending object, it's removed from the pending queue
        if science_case and science_case != "all":
            try:
                remove_pending_vote(db, ztfid, science_case)
            except Exception as e:
                logger.warning(f"Could not remove pending vote for {ztfid} from {science_case}: {e}")
    
    # Handle tags from metadata in payload (only if adding a vote, not removing)
    if vote_action == "added":
        tags_added_manually = False
        if vote_payload_details and "tags" in vote_payload_details and vote_payload_details["tags"]:
            tags_added_manually = True
            # Delete existing tags
            db.query(models.Tag).filter(
                models.Tag.user_id == current_user.id,
                models.Tag.ztfid == ztfid
            ).delete()
            
            # Add new tags
            for tag_name in vote_payload_details["tags"]:
                new_tag = models.Tag(
                    user_id=current_user.id,
                    ztfid=ztfid,
                    tag_name=tag_name
                )
                db.add(new_tag)

        # Handle notes from metadata in payload
        notes_added_manually = False
        if vote_payload_details and "notes" in vote_payload_details and vote_payload_details["notes"]:
            notes_added_manually = True
            # Check if note exists
            existing_note = db.query(models.Note).filter(
                models.Note.user_id == current_user.id,
                models.Note.ztfid == ztfid
            ).first()
            
            if existing_note:
                # Update existing note
                existing_note.text = vote_payload_details["notes"]
            else:
                # Create new note
                new_note = models.Note(
                    user_id=current_user.id,
                    ztfid=ztfid,
                    text=vote_payload_details["notes"]
                )
                db.add(new_note)
        
        # Default tagging with science interests if no manual tags or notes were added
        has_explicit_tags = bool(vote_payload_details and "tags" in vote_payload_details and vote_payload_details["tags"])
        has_explicit_notes = bool(vote_payload_details and "notes" in vote_payload_details and vote_payload_details["notes"])

        if not has_explicit_tags and not has_explicit_notes and current_user.science_interests:
            # Delete existing tags first to avoid duplicates
            db.query(models.Tag).filter(
                models.Tag.user_id == current_user.id,
                models.Tag.ztfid == ztfid
            ).delete()
            for interest in current_user.science_interests:
                default_tag = models.Tag(
                    user_id=current_user.id,
                    ztfid=ztfid,
                    tag_name=str(interest)
                )
                db.add(default_tag)
            logger.info(f"Applied default science interest tags for user {current_user.id}, ZTFID {ztfid}: {current_user.science_interests}")

    db.commit()
    return {"status": "success", "action": vote_action, "vote_type": vote_type}

@api_app.post("/skip")
async def skip_object(
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Handle skipped objects."""
    data = await request.json()
    ztfid = data.get("ztfid")
    
    if not ztfid:
        raise HTTPException(status_code=400, detail="Missing ztfid")
    
    # Check if skip vote already exists specifically
    existing_skip_vote = db.query(models.Vote).filter(
        models.Vote.user_id == current_user.id,
        models.Vote.ztfid == ztfid,
        models.Vote.vote_type == "skip"
    ).first()
    
    if existing_skip_vote:
        # Skip vote already exists, just update timestamp
        existing_skip_vote.last_updated = datetime.utcnow()
        logger.info(f"Updated existing skip vote for {ztfid} by user {current_user.id}")
    else:
        # Create new skip vote
        new_vote = models.Vote(
            user_id=current_user.id,
            ztfid=ztfid,
            vote_type="skip",
            science_case=""
        )
        db.add(new_vote)
        logger.info(f"Created new skip vote for {ztfid} by user {current_user.id}")
    
    db.commit()
    return {"status": "success"}

@api_app.get("/targets")
async def get_targets(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the list of targets with their details."""
    # Get target votes
    target_votes = db.query(models.Vote).filter(
        models.Vote.user_id == current_user.id,
        models.Vote.vote_type == "target"
    ).all()
    
    if not target_votes:
        return []
    
    targets = []
    for vote in target_votes:
        # Get object details from feature bank
        feature = db.query(models.FeatureBank).filter(
            models.FeatureBank.ztfid == vote.ztfid
        ).first()
        
        if feature:
            target_details = {
                "ztfid": vote.ztfid,
                "ra": feature.ra,
                "dec": feature.dec,
                "latest_magnitude": feature.latest_magnitude,
                "created_at": vote.created_at
            }
            targets.append(target_details)
        else:
            # If no feature details in DB, try to get from recommender's feature bank
            ra = None
            dec = None
            latest_magnitude = None
            
            if vote.vote_details and isinstance(vote.vote_details, dict):
                object_details = vote.vote_details.get("object_details", {})
                if object_details:
                    ra = object_details.get("ra")
                    dec = object_details.get("dec")
                    latest_magnitude = object_details.get("latest_magnitude")
            
            # If still no coordinates, try to get from recommender's feature bank
            if ra is None or dec is None:
                if recommender_engine.feature_bank is not None:
                    obj_row = recommender_engine.feature_bank[recommender_engine.feature_bank['ZTFID'] == vote.ztfid]
                    if not obj_row.empty:
                        # Convert numpy types to regular Python types to avoid JSON serialization issues
                        row = obj_row.iloc[0]
                        ra = float(row.get('ra')) if row.get('ra') is not None and not pd.isna(row.get('ra')) else None
                        dec = float(row.get('dec')) if row.get('dec') is not None and not pd.isna(row.get('dec')) else None
                        if 'latest_magnitude' in obj_row.columns:
                            latest_magnitude = float(row.get('latest_magnitude')) if row.get('latest_magnitude') is not None and not pd.isna(row.get('latest_magnitude')) else None
                        logger.info(f"Retrieved coordinates for {vote.ztfid} from recommender feature bank: RA={ra}, Dec={dec}")
            
            # Ensure all values are JSON serializable
            target_details = {
                "ztfid": vote.ztfid,
                "ra": float(ra) if ra is not None and not pd.isna(ra) else None,
                "dec": float(dec) if dec is not None and not pd.isna(dec) else None,
                "latest_magnitude": float(latest_magnitude) if latest_magnitude is not None and not pd.isna(latest_magnitude) else None,
                "created_at": vote.created_at
            }
            targets.append(target_details)
    
    return targets

@api_app.post("/remove-target")
async def remove_target(
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove a target from the target list."""
    try:
        logger.info(f"Remove target request from user: {current_user.username}")
        
        data = await request.json()
        ztfid = data.get("ztfid")
        
        logger.info(f"Attempting to remove target: {ztfid}")
        
        if not ztfid:
            logger.warning("Missing ztfid in remove target request")
            raise HTTPException(status_code=400, detail="Missing ztfid")
        
        # Find the target vote
        target_vote = db.query(models.Vote).filter(
            models.Vote.user_id == current_user.id,
            models.Vote.ztfid == ztfid,
            models.Vote.vote_type == "target"
        ).first()
        
        if target_vote:
            logger.info(f"Found target vote for {ztfid}, deleting it")
            # Delete the target vote (don't change to like - user may already have a like vote)
            db.delete(target_vote)
            db.commit()
            logger.info(f"Successfully removed target vote for {ztfid}")
        else:
            logger.info(f"No target vote found for {ztfid} by user {current_user.username}")
        
        return {"status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in remove_target: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_app.get("/tags/{ztfid}")
async def get_tags(
    ztfid: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get tags for an object, organized by category."""
    tags = db.query(models.Tag).filter(
        models.Tag.user_id == current_user.id,
        models.Tag.ztfid == ztfid
    ).all()
    
    # Organize tags by category
    tags_by_category = {
        "science": [],
        "spectra": [],
        "photometry": [],
        "host": [],
        "general": []
    }
    
    for tag in tags:
        category = tag.category or "general"
        if category in tags_by_category:
            tags_by_category[category].append(tag.tag_name)
        else:
            tags_by_category["general"].append(tag.tag_name)
    
    return tags_by_category

@api_app.post("/tags/{ztfid}")
async def save_tags(
    ztfid: str,
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save tags for an object, organized by category."""
    data = await request.json()
    
    # Handle both old format (simple list) and new format (categorized)
    if isinstance(data.get("tags"), list):
        # Old format - treat as general tags
        tags_by_category = {"general": data.get("tags", [])}
    else:
        # New format - categorized tags
        tags_by_category = data.get("tags", {})
    
    # Delete existing tags for this object
    db.query(models.Tag).filter(
        models.Tag.user_id == current_user.id,
        models.Tag.ztfid == ztfid
    ).delete()
    
    # Add new tags by category
    for category, tag_list in tags_by_category.items():
        for tag_name in tag_list:
            if tag_name.strip():  # Only add non-empty tags
                new_tag = models.Tag(
                    user_id=current_user.id,
                    ztfid=ztfid,
                    tag_name=tag_name.strip(),
                    category=category
                )
                db.add(new_tag)
    
    db.commit()
    return {"status": "success"}

@api_app.get("/notes/{ztfid}")
async def get_notes(
    ztfid: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get notes for an object."""
    note = db.query(models.Note).filter(
        models.Note.user_id == current_user.id,
        models.Note.ztfid == ztfid
    ).first()
    
    return {"text": note.text if note else ""}

@api_app.post("/notes/{ztfid}")
async def save_notes(
    ztfid: str,
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save notes for an object."""
    data = await request.json()
    text = data.get("text", "")
    
    # Check if note exists
    note = db.query(models.Note).filter(
        models.Note.user_id == current_user.id,
        models.Note.ztfid == ztfid
    ).first()
    
    if note:
        # Update existing note
        note.text = text
    else:
        # Create new note
        new_note = models.Note(
            user_id=current_user.id,
            ztfid=ztfid,
            text=text
        )
        db.add(new_note)
    
    db.commit()
    return {"status": "success"}

@api_app.post("/audio_note/{ztfid}")
async def save_audio_note(
    ztfid: str,
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(None), # Get session_id from form data
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save an audio note for a given ZTF ID."""
    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid audio file type.")

    audio_data = await audio_file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # Attempt transcription
    transcription = None
    try:
        # Convert audio to WAV using pydub (in-memory)
        # pydub will attempt to infer format from filename or content_type
        # Forcing webm if content_type is missing, common for browser MediaRecorder
        audio_format = audio_file.content_type.split('/')[-1] if audio_file.content_type else 'webm'
        
        # Ensure common browser audio formats are handled for pydub
        if audio_format == 'ogg': # ogg often contains opus or vorbis
             audio_format = 'ogg' 
        elif audio_format == 'mp4': # mp4 audio
             audio_format = 'm4a' # pydub uses m4a for mp4 audio typically
        elif audio_format not in ['wav', 'mp3', 'flac', 'webm', 'ogg', 'm4a']:
            logger.warning(f"Potentially unsupported audio format for pydub conversion: {audio_format}. Trying with 'webm'.")
            audio_format = 'webm' # Default to webm if truly unknown/exotic

        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format)
        
        # Export to a temporary WAV file in memory for SpeechRecognition
        wav_audio_io = io.BytesIO()
        audio_segment.export(wav_audio_io, format="wav")
        wav_audio_io.seek(0) # Rewind to the beginning of the stream

        r = sr.Recognizer()
        with sr.AudioFile(wav_audio_io) as source:
            audio_for_recognition = r.record(source)  # read the entire audio file
        
        # Recognize speech using Google Web Speech API (requires internet)
        # This is the default recognizer if no specific one is chosen and an API key isn't set
        # It has limitations but is free for low volume.
        transcription = r.recognize_google(audio_for_recognition)
        logger.info(f"Transcription successful for {audio_file.filename}: {transcription}")
    except sr.UnknownValueError:
        logger.warning(f"Google Web Speech API could not understand audio for {audio_file.filename}")
        transcription = "[Audio not recognized]"
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Web Speech API for {audio_file.filename}; {e}")
        transcription = "[Transcription service error]"
    except Exception as e:
        logger.error(f"Error during audio transcription for {audio_file.filename}: {e}")
        transcription = "[Transcription failed]"

    new_audio_note = models.AudioNote(
        user_id=current_user.id,
        ztfid=ztfid,
        session_id=session_id,
        audio_data=audio_data,
        filename=audio_file.filename,
        content_type=audio_file.content_type,
        timestamp=datetime.utcnow(), # Explicitly set timestamp
        transcription=transcription # Add transcription here
    )

    try:
        db.add(new_audio_note)
        db.commit()
        db.refresh(new_audio_note)
        logger.info(f"Audio note saved for user {current_user.id}, ZTFID {ztfid}, filename {audio_file.filename}")
        return {"status": "success", "note_id": new_audio_note.id, "filename": audio_file.filename, "transcription": transcription}
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving audio note for user {current_user.id}, ZTFID {ztfid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save audio note.")

@api_app.get("/vote-counts/{ztfid}")
async def get_vote_counts(
    ztfid: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get vote counts for a specific ZTFID."""
    try:
        # Get all votes for this ZTFID
        votes = db.query(models.Vote).filter(models.Vote.ztfid == ztfid).all()
        
        # Count votes by type
        counts = {}
        for vote in votes:
            vote_type = vote.vote_type
            counts[vote_type] = counts.get(vote_type, 0) + 1
        
        # Get current user's vote
        user_vote = db.query(models.Vote).filter(
            models.Vote.user_id == current_user.id,
            models.Vote.ztfid == ztfid
        ).first()
        
        result = {
            "counts": counts,
            "user_vote": user_vote.vote_type if user_vote else None
        }
        return result
    except Exception as e:
        logger.error(f"Vote counts endpoint error for {ztfid}: {e}", exc_info=True)
        raise



@api_app.get("/audio_notes/{ztfid}")
async def get_audio_notes(
    ztfid: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all audio notes for a given ZTF ID from all users."""
    try:
        audio_notes = db.query(models.AudioNote).join(models.User).filter(
            models.AudioNote.ztfid == ztfid
        ).order_by(models.AudioNote.timestamp.desc()).all()
        
        result = [
            {
                "id": note.id,
                "ztfid": note.ztfid,
                "filename": note.filename,
                "created_at": note.timestamp.isoformat(),
                "transcription": note.transcription,
                "username": note.user.username,
                "user_id": note.user_id,
                "is_own": note.user_id == current_user.id
            }
            for note in audio_notes
        ]
        return result
    except Exception as e:
        logger.error(f"Audio notes endpoint error for {ztfid}: {e}", exc_info=True)
        raise



@api_app.get("/audio_notes/file/{filename}")
async def get_audio_file(
    filename: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Serve an audio file."""
    audio_note = db.query(models.AudioNote).filter(
        models.AudioNote.filename == filename
    ).first()
    
    if not audio_note:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return Response(
        content=audio_note.audio_data,
        media_type=audio_note.content_type or "audio/webm"
    )

@api_app.delete("/audio_notes/{note_id}")
async def delete_audio_note(
    note_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an audio note."""
    audio_note = db.query(models.AudioNote).filter(
        models.AudioNote.id == note_id,
        models.AudioNote.user_id == current_user.id
    ).first()
    
    if not audio_note:
        raise HTTPException(status_code=404, detail="Audio note not found")
    
    db.delete(audio_note)
    db.commit()
    
    return {"status": "success", "message": "Audio note deleted"}

@api_app.post("/generate-finders")
async def generate_finders(
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate finder charts for all targets and return as zip download."""
    import subprocess
    from pathlib import Path
    import os
    import asyncio
    import time
    import zipfile
    import tempfile
    from fastapi.responses import FileResponse
    
    logger.info(f"Generate finders called by user: {current_user.username}")
    
    # Get target votes
    target_votes = db.query(models.Vote).filter(
        models.Vote.user_id == current_user.id,
        models.Vote.vote_type == "target"
    ).all()
    
    if not target_votes:
        logger.info(f"No targets found for user {current_user.username}")
        return {"status": "error", "message": "No targets found"}
    
    logger.info(f"Found {len(target_votes)} targets for user {current_user.username}")
    
    # Create temporary directory for this generation session
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        finder_charts_dir = temp_path / "finder_charts"
        finder_charts_dir.mkdir(exist_ok=True)
        
        successful_completions = 0
        failed_objects = []
        successful_objects = []
        successful_files = []
        process_results = []
        
        for vote in target_votes:
            # Get object details from feature bank
            feature = db.query(models.FeatureBank).filter(
                models.FeatureBank.ztfid == vote.ztfid
            ).first()
            
            if feature and feature.ra is not None and feature.dec is not None:
                try:
                    output_file = finder_charts_dir / f"{vote.ztfid}_finder.png"
                    cmd = [
                        "python",
                        str(Path.cwd() / "finder_charts" / "mkFinderChart_fixed.py"),
                        "-s", vote.ztfid,
                        "-r", f"{feature.ra:.5f}",
                        "-d", f"{feature.dec:.5f}",
                        "-o", str(output_file)
                    ]
                    
                    logger.info(f"Starting finder chart generation for {vote.ztfid} (RA={feature.ra:.5f}, Dec={feature.dec:.5f})")
                    
                    # Run process and wait for completion (with timeout)
                    start_time = time.time()
                    try:
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=60,  # 60 second timeout per object
                            text=True,
                            cwd=str(finder_charts_dir)
                        )
                        
                        execution_time = time.time() - start_time
                        
                        # Check if the output file was created
                        if output_file.exists() and result.returncode == 0:
                            successful_completions += 1
                            successful_objects.append(vote.ztfid)
                            successful_files.append(output_file)
                            logger.info(f"Successfully generated finder chart for {vote.ztfid} in {execution_time:.1f}s")
                            process_results.append({
                                "ztfid": vote.ztfid,
                                "status": "success",
                                "output_file": str(output_file),
                                "execution_time": execution_time
                            })
                        else:
                            # Process completed but failed
                            error_msg = f"Process failed (exit code: {result.returncode})"
                            if result.stderr:
                                error_msg += f" - {result.stderr.strip()}"
                            failed_objects.append(f"{vote.ztfid} ({error_msg})")
                            logger.error(f"Finder chart generation failed for {vote.ztfid}: {error_msg}")
                            process_results.append({
                                "ztfid": vote.ztfid,
                                "status": "failed",
                                "error": error_msg,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                                "execution_time": execution_time
                            })
                    
                    except subprocess.TimeoutExpired:
                        execution_time = time.time() - start_time
                        failed_objects.append(f"{vote.ztfid} (timeout after {execution_time:.1f}s)")
                        logger.error(f"Finder chart generation timed out for {vote.ztfid} after {execution_time:.1f}s")
                        process_results.append({
                            "ztfid": vote.ztfid,
                            "status": "timeout",
                            "execution_time": execution_time
                        })
                    
                except Exception as e:
                    logger.error(f"Failed to start finder chart generation for {vote.ztfid}: {e}")
                    failed_objects.append(f"{vote.ztfid} (startup error: {str(e)})")
                    process_results.append({
                        "ztfid": vote.ztfid,
                        "status": "startup_error",
                        "error": str(e)
                    })
            else:
                logger.warning(f"Missing coordinates for {vote.ztfid}")
                failed_objects.append(f"{vote.ztfid} (missing coordinates)")
                process_results.append({
                    "ztfid": vote.ztfid,
                    "status": "missing_coordinates"
                })
        
        # If we have any successful files, create a zip
        if successful_files:
            # Create zip file
            zip_filename = f"finder_charts_{current_user.username}_{int(time.time())}.zip"
            zip_path = temp_path / zip_filename
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in successful_files:
                    # Add file to zip with just its name (not full path)
                    zipf.write(file_path, file_path.name)
                
                # Create summary text file
                summary_content = f"Finder Charts Generation Summary\n"
                summary_content += f"="*50 + "\n\n"
                summary_content += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                summary_content += f"User: {current_user.username}\n"
                summary_content += f"Total targets: {len(target_votes)}\n"
                summary_content += f"Successful: {successful_completions}\n"
                summary_content += f"Failed: {len(failed_objects)}\n\n"
                
                if successful_objects:
                    summary_content += f"Successful Objects:\n"
                    for obj in successful_objects:
                        summary_content += f"  - {obj}\n"
                    summary_content += "\n"
                
                if failed_objects:
                    summary_content += f"Failed Objects:\n"
                    for obj in failed_objects:
                        summary_content += f"  - {obj}\n"
                
                # Add summary to zip
                zipf.writestr("summary.txt", summary_content)
            
            logger.info(f"Created zip file with {len(successful_files)} finder charts")
            
            # Copy zip to a permanent location for download
            permanent_zip_dir = Path.cwd() / "finder_charts" / "downloads"
            permanent_zip_dir.mkdir(exist_ok=True)
            permanent_zip_path = permanent_zip_dir / zip_filename
            
            import shutil
            shutil.copy2(zip_path, permanent_zip_path)
            
            # Return file response for download
            return FileResponse(
                path=str(permanent_zip_path),
                filename=zip_filename,
                media_type='application/zip',
                headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
            )
        
        else:
            # No successful files, return error
            total_targets = len(target_votes)
            message = f"Failed to generate any finder charts from {total_targets} targets"
            
            if failed_objects:
                message += f"\n\nFailed objects:\n" + "\n".join(f"- {obj}" for obj in failed_objects)
            
            logger.info(f"Generate finders summary: 0 successful, {len(failed_objects)} failed")
            
            return {
                "status": "failed",
                "message": message,
                "total_targets": total_targets,
                "successful_count": 0,
                "failed_count": len(failed_objects),
                "failed_objects": failed_objects,
                "detailed_results": process_results
            }

@api_app.get("/stats")
async def get_stats(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user statistics."""
    try:
        likes_count = db.query(func.count(models.Vote.id)).filter(
            models.Vote.user_id == current_user.id,
            models.Vote.vote_type == "like"
        ).scalar()
        
        dislikes_count = db.query(func.count(models.Vote.id)).filter(
            models.Vote.user_id == current_user.id,
            models.Vote.vote_type == "dislike"
        ).scalar()
        
        targets_count = db.query(func.count(models.Vote.id)).filter(
            models.Vote.user_id == current_user.id,
            models.Vote.vote_type == "target"
        ).scalar()
        
        skips_count = db.query(func.count(models.Vote.id)).filter(
            models.Vote.user_id == current_user.id,
            models.Vote.vote_type == "skip"
        ).scalar()
        
        tags_count = db.query(func.count(models.Tag.id)).filter(
            models.Tag.user_id == current_user.id
        ).scalar()
        
        notes_count = db.query(func.count(models.Note.id)).filter(
            models.Note.user_id == current_user.id
        ).scalar()
        
        result = {
            "likes": likes_count,
            "dislikes": dislikes_count,
            "targets": targets_count,
            "skips": skips_count,
            "tags": tags_count,
            "notes": notes_count
        }
        return result
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}", exc_info=True)
        raise



@api_app.get("/user/profile")
async def get_user_profile(
    current_user: models.User = Depends(get_current_user)
):
    """Get user profile information."""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "data_sharing_consent": current_user.data_sharing_consent,
        "is_admin": current_user.is_admin,
        "science_interests": current_user.science_interests
    }



@api_app.post("/user/profile")
async def update_user_profile(
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile."""
    data = await request.json()
    
    if "email" in data:
        current_user.email = data["email"]
    if "data_sharing_consent" in data:
        current_user.data_sharing_consent = data["data_sharing_consent"]
    if "science_interests" in data:
        current_user.science_interests = data["science_interests"]
        
    db.commit()
    db.refresh(current_user)
    return {"message": "Profile updated successfully"}

@api_app.post("/update-feature-bank")
async def update_feature_bank(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update the feature bank from the CSV file."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to perform this action",
        )
    
    recommender_engine.update_feature_bank(db)
    return {"status": "success"}

def run_feature_extraction_background(lookback_days: float, force_reprocess: bool):
    """Run feature extraction in the background."""
    try:
        logger.info(f"Starting background feature extraction with lookback_days={lookback_days}, force_reprocess={force_reprocess}")
        
        # Check if Antares is available and working
        try:
            import antares_client
            from antares_client.search import search
            test_mode = False
            logger.info("Antares available for background extraction")
        except ImportError as e:
            test_mode = True
            logger.warning(f"Antares not available - using test mode for background extraction: {e}")
        except Exception as e:
            test_mode = True
            logger.warning(f"Antares import failed - using test mode for background extraction: {e}")
        
        # Get a new database session for the background task
        db = SessionLocal()
        
        try:
            extraction_run = extract_features_for_recent_objects(
                db, lookback_days, force_reprocess, test_mode=test_mode
            )
            logger.info(f"Background feature extraction completed successfully: run_id={extraction_run.id}, objects_processed={extraction_run.objects_processed}")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Background feature extraction failed: {e}", exc_info=True)

@api_app.post("/extract-features")
async def extract_features(
    background_tasks: BackgroundTasks,
    lookback_days: float = 20.0,
    force_reprocess: bool = False,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manually trigger feature extraction."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to perform this action",
        )
    
    # CRITICAL: Check if feature extraction is already running
    last_run = get_last_extraction_run(db)
    if last_run and last_run.status == "running":
        # Calculate how long it's been running
        from astropy.time import Time
        hours_running = (Time.now().mjd - last_run.mjd_run) * 24
        
        # If it's been running for more than 2 hours, consider it stuck and allow override
        if hours_running > 2.0:
            logger.warning(f"Feature extraction appears stuck (running {hours_running:.1f} hours), allowing override")
            last_run.status = "failed"
            last_run.error_message = f"Extraction stuck after {hours_running:.1f} hours, manually overridden"
            last_run.completed_at = datetime.utcnow()
            db.commit()
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Feature extraction is already running (started {hours_running:.1f} hours ago). Please wait for it to complete or contact an administrator if it appears stuck."
            )
    
    try:
        # Start extraction in background
        background_tasks.add_task(
            run_feature_extraction_background,
            lookback_days,
            force_reprocess
        )
        
        logger.info(f"Feature extraction started in background by {current_user.username}")
        
        return {
            "status": "started",
            "message": "Feature extraction started in background",
            "started_by": current_user.username
        }
    except Exception as e:
        logger.error(f"Failed to start feature extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/extraction-status")
async def get_extraction_status(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the status of the last feature extraction run."""
    last_run = get_last_extraction_run(db)
    
    if not last_run:
        return {"status": "never_run"}
    
    # Calculate runtime for running extractions
    runtime_info = {}
    if last_run.status == "running":
        from astropy.time import Time
        runtime_hours = float((Time.now().mjd - last_run.mjd_run) * 24)
        runtime_info = {
            "runtime_hours": runtime_hours,
            "is_stuck": bool(runtime_hours > 2.0)  # Consider stuck after 2 hours
        }
    
    return {
        "status": last_run.status,
        "run_date": last_run.run_date.isoformat() if last_run.run_date else None,
        "mjd_run": float(last_run.mjd_run) if last_run.mjd_run is not None else None,
        "lookback_days": float(last_run.lookback_days) if last_run.lookback_days is not None else None,
        "objects_found": int(last_run.objects_found) if last_run.objects_found is not None else None,
        "objects_processed": int(last_run.objects_processed) if last_run.objects_processed is not None else None,
        "processing_time_seconds": float(last_run.processing_time_seconds) if last_run.processing_time_seconds is not None else None,
        "error_message": last_run.error_message,
        **runtime_info
    }

@api_app.get("/anomaly-notifications")
async def get_anomaly_notifications(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get unacknowledged anomaly notifications."""
    try:
        notifications = anomaly_service.get_unacknowledged_notifications(db)
        
        result = []
        for notification in notifications:
            result.append({
                "id": notification.id,
                "detection_run_id": notification.detection_run_id,
                "objects_detected": notification.objects_detected,
                "ztfids_detected": notification.ztfids_detected or [],
                "created_at": notification.created_at.isoformat(),
                "acknowledged": notification.acknowledged
            })
        
        return {"notifications": result}
        
    except Exception as e:
        logger.error(f"Error getting anomaly notifications: {e}")
        return {"notifications": []}

@api_app.post("/anomaly-notifications/{notification_id}/acknowledge")
async def acknowledge_anomaly_notification(
    notification_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Acknowledge an anomaly notification."""
    try:
        anomaly_service.acknowledge_notification(db, notification_id)
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error acknowledging notification {notification_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error acknowledging notification: {str(e)}")

@api_app.get("/anomaly-results/{ztfid}")
async def get_anomaly_result(
    ztfid: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get anomaly detection result for a specific object."""
    try:
        result = db.query(models.AnomalyDetectionResult).filter(
            models.AnomalyDetectionResult.ztfid == ztfid
        ).first()
        
        if not result:
            return {"anomaly_result": None}
        
        return {
            "anomaly_result": {
                "ztfid": result.ztfid,
                "anomaly_score": result.anomaly_score,
                "mjd_anom": result.mjd_anom,
                "anom_scores": result.anom_scores,
                "norm_scores": result.norm_scores,
                "detection_threshold": result.detection_threshold,
                "is_anomalous": result.is_anomalous,
                "created_at": result.created_at.isoformat(),
                "updated_at": result.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting anomaly result for {ztfid}: {e}")
        return {"anomaly_result": None}

@api_app.get("/pending-objects")
async def get_pending_objects_summary(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get summary of pending objects for all science cases."""
    try:
        # Define all possible science cases
        science_cases = ["anomalous", "snia-like", "ccsn-like", "long-lived", "precursor"]
        
        pending_summary = {}
        total_pending = 0
        
        for science_case in science_cases:
            pending_objects = get_pending_objects_for_science_case(db, science_case)
            pending_summary[science_case] = len(pending_objects)
            total_pending += len(pending_objects)
        
        return {
            "total_pending": total_pending,
            "by_science_case": pending_summary,
            "available_science_cases": science_cases
        }
        
    except Exception as e:
        logger.error(f"Error getting pending objects summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_app.get("/pending-objects/{science_case}")
async def get_pending_objects_for_science_case_endpoint(
    science_case: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed list of pending objects for a specific science case."""
    try:
        pending_ztfids = get_pending_objects_for_science_case(db, science_case)
        
        if not pending_ztfids:
            return {
                "science_case": science_case,
                "count": 0,
                "objects": []
            }
        
        # Get details for these objects from the feature bank
        pending_objects = []
        for ztfid in pending_ztfids:
            feature = db.query(models.FeatureBank).filter(
                models.FeatureBank.ztfid == ztfid
            ).first()
            
            obj_details = {
                "ztfid": ztfid,
                "ra": feature.ra if feature else None,
                "dec": feature.dec if feature else None,
                "latest_magnitude": feature.latest_magnitude if feature else None,
                "mjd_extracted": feature.mjd_extracted if feature else None
            }
            
            # Add anomaly score if this is for anomalous science case
            if science_case == "anomalous":
                anomaly_result = db.query(models.AnomalyDetectionResult).filter(
                    models.AnomalyDetectionResult.ztfid == ztfid
                ).first()
                if anomaly_result:
                    obj_details["anomaly_score"] = anomaly_result.anomaly_score
                    obj_details["is_anomalous"] = anomaly_result.is_anomalous
            
            pending_objects.append(obj_details)
        
        return {
            "science_case": science_case,
            "count": len(pending_objects),
            "objects": pending_objects
        }
        
    except Exception as e:
        logger.error(f"Error getting pending objects for {science_case}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



@api_app.get("/comments/{ztfid}")
async def get_comments(
    ztfid: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all comments for an object from all users."""
    try:
        comments = db.query(models.Comment).filter(
            models.Comment.ztfid == ztfid
        ).order_by(models.Comment.created_at.asc()).all()
        
        comment_list = []
        for comment in comments:
            comment_list.append({
                "id": comment.id,
                "text": comment.text,
                "username": comment.user.username,
                "user_id": comment.user_id,
                "created_at": comment.created_at.isoformat(),
                "updated_at": comment.updated_at.isoformat(),
                "is_own": comment.user_id == current_user.id
            })
        
        return comment_list
    except Exception as e:
        logger.error(f"Comments endpoint error for {ztfid}: {e}", exc_info=True)
        raise



@api_app.post("/comments/{ztfid}")
async def create_comment(
    ztfid: str,
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new comment for an object."""
    data = await request.json()
    text = data.get("text", "").strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Comment text cannot be empty")
    
    new_comment = models.Comment(
        user_id=current_user.id,
        ztfid=ztfid,
        text=text
    )
    db.add(new_comment)
    db.commit()
    db.refresh(new_comment)
    
    return {
        "id": new_comment.id,
        "text": new_comment.text,
        "username": current_user.username,
        "user_id": current_user.id,
        "created_at": new_comment.created_at.isoformat(),
        "updated_at": new_comment.updated_at.isoformat(),
        "is_own": True
    }



@api_app.put("/comments/{comment_id}")
async def update_comment(
    comment_id: int,
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a comment (only by the comment author)."""
    data = await request.json()
    text = data.get("text", "").strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Comment text cannot be empty")
    
    comment = db.query(models.Comment).filter(
        models.Comment.id == comment_id,
        models.Comment.user_id == current_user.id
    ).first()
    
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found or not authorized")
    
    comment.text = text
    db.commit()
    db.refresh(comment)
    
    return {
        "id": comment.id,
        "text": comment.text,
        "username": current_user.username,
        "user_id": current_user.id,
        "created_at": comment.created_at.isoformat(),
        "updated_at": comment.updated_at.isoformat(),
        "is_own": True
    }



@api_app.delete("/comments/{comment_id}")
async def delete_comment(
    comment_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a comment (only by the comment author)."""
    comment = db.query(models.Comment).filter(
        models.Comment.id == comment_id,
        models.Comment.user_id == current_user.id
    ).first()
    
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found or not authorized")
    
    db.delete(comment)
    db.commit()
    
    return {"status": "success"}



@api_app.get("/history")
async def get_user_history(
    filter: str = "all",
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive user interaction history."""
    history_items = []
    statistics = {
        "like": 0,
        "dislike": 0,
        "target": 0,
        "skip": 0,
        "comment": 0,
        "audio": 0,
        "tag": 0
    }
    
    # Get votes
    votes = db.query(models.Vote).filter(models.Vote.user_id == current_user.id).all()
    for vote in votes:
        if filter == "all" or filter == vote.vote_type:
            history_items.append({
                "type": vote.vote_type,
                "ztfid": vote.ztfid,
                "timestamp": vote.created_at.isoformat(),
                "science_case": vote.science_case,
                "content": None
            })
        statistics[vote.vote_type] = statistics.get(vote.vote_type, 0) + 1
    
    # Get comments
    comments = db.query(models.Comment).filter(models.Comment.user_id == current_user.id).all()
    for comment in comments:
        if filter == "all" or filter == "comment":
            history_items.append({
                "type": "comment",
                "ztfid": comment.ztfid,
                "timestamp": comment.created_at.isoformat(),
                "science_case": None,
                "content": comment.text[:100] + "..." if len(comment.text) > 100 else comment.text
            })
    statistics["comment"] = len(comments)
    
    # Get audio notes
    audio_notes = db.query(models.AudioNote).filter(models.AudioNote.user_id == current_user.id).all()
    for audio_note in audio_notes:
        if filter == "all" or filter == "audio":
            history_items.append({
                "type": "audio",
                "ztfid": audio_note.ztfid,
                "timestamp": audio_note.created_at.isoformat(),
                "science_case": None,
                "content": None
            })
    statistics["audio"] = len(audio_notes)
    
    # Get tags (group by ztfid)
    tags = db.query(models.Tag).filter(models.Tag.user_id == current_user.id).all()
    tag_groups = {}
    for tag in tags:
        if tag.ztfid not in tag_groups:
            tag_groups[tag.ztfid] = {
                "ztfid": tag.ztfid,
                "timestamp": tag.created_at.isoformat(),
                "tags": []
            }
        tag_groups[tag.ztfid]["tags"].append(tag.tag_name)
    
    for ztfid, tag_data in tag_groups.items():
        if filter == "all" or filter == "tag":
            history_items.append({
                "type": "tag",
                "ztfid": ztfid,
                "timestamp": tag_data["timestamp"],
                "science_case": None,
                "content": ", ".join(tag_data["tags"])
            })
    statistics["tag"] = len(tag_groups)
    
    # Sort by timestamp (most recent first)
    history_items.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "history": history_items,
        "statistics": statistics
    }



@api_app.get("/spectra/{ztfid}")
async def get_spectra(
    ztfid: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get available spectra from WiseREP for a ZTFID using the proven API approach."""
    import requests
    import zipfile
    import io
    import csv
    import tempfile
    import os
    import json
    import datetime
    
    logger.info(f"Starting spectrum search for {ztfid} requested by user {current_user.username}")
    
    try:
        # First get the object's coordinates from our database
        feature = db.query(models.FeatureBank).filter(
            models.FeatureBank.ztfid == ztfid
        ).first()
        
        if not feature or feature.ra is None or feature.dec is None:
            logger.warning(f"No coordinates found for {ztfid} in database - feature exists: {feature is not None}")
            if feature:
                logger.warning(f"Feature found but coordinates missing: RA={feature.ra}, Dec={feature.dec}")
            return {
                'ztfid': ztfid,
                'spectra': [],
                'total_count': 0,
                'source_url': f"https://www.wiserep.org/search/spectra?search_term={ztfid}",
                'message': f'No coordinates available for {ztfid} in feature bank',
                'error': 'missing_coordinates'
            }
        
        ra = feature.ra
        dec = feature.dec
        logger.info(f"Found coordinates for {ztfid}: RA={ra:.6f}, Dec={dec:.6f}")
        
        # WiseREP API configuration using proven approach
        WISEREP = "www.wiserep.org"
        url_wis_spectra_search = f"https://{WISEREP}/search/spectra"
        
        # Use the working API key from the provided code
        personal_api_key = "0098c283eb37f9c6aaa98a021db168fe0cdf83d1"
        
        # User agent configuration (proven format)
        WIS_USER_NAME = "agagliano"
        WIS_USER_ID = "api_user"
        
        # Search parameters using coordinates with small radius around target
        search_radius_deg = 0.01  # ~36 arcseconds radius
        ra_min = ra - search_radius_deg
        ra_max = ra + search_radius_deg
        dec_min = dec - search_radius_deg
        dec_max = dec + search_radius_deg
        
        query_params = f"&public=yes&ra_range_min={ra_min:.6f}&ra_range_max={ra_max:.6f}&decl_range_min={dec_min:.6f}&decl_range_max={dec_max:.6f}"
        download_params = "&num_page=50&format=csv&files_type=ascii"
        parameters = "?" + query_params + download_params + "&personal_api_key=" + personal_api_key
        
        logger.info(f"WiseREP search parameters: RA {ra_min:.6f} to {ra_max:.6f}, Dec {dec_min:.6f} to {dec_max:.6f}")
        
        # User agent marker (exact format from working code)
        wis_marker = f'wis_marker{{"wis_id": "{WIS_USER_ID}", "type": "user", "name": "{WIS_USER_NAME}"}}'
        headers = {'User-Agent': wis_marker}
        
        # Collect all spectra across pages
        all_spectra = []
        page_num = 0
        max_pages = 5  # Reasonable limit to prevent infinite loops
        
        while page_num < max_pages:
            # URL for this page
            url = url_wis_spectra_search + parameters + f"&page={page_num}"
            
            # Make POST request (as per working code)
            response = requests.post(url, headers=headers, stream=True, timeout=30)
            
            # Check for end of pages (404 means no more data)
            if response.status_code == 404:
                logger.info(f"No more pages available after page {page_num}")
                break
            
            # Check for other errors
            if response.status_code != 200:
                logger.warning(f"Page {page_num + 1} failed with status {response.status_code}")
                if page_num == 0:
                    # If first page fails, return error
                    return {
                        'ztfid': ztfid,
                        'spectra': [],
                        'total_count': 0,
                        'source_url': f"https://www.wiserep.org/search/spectra",
                        'message': f'WiseREP API error: {response.status_code}',
                        'error': 'api_error'
                    }
                else:
                    # If later page fails, just stop pagination
                    break
            
            # Process the ZIP response
            try:
                # Create temporary file for ZIP
                with tempfile.NamedTemporaryFile() as temp_zip:
                    # Download ZIP content
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_zip.write(chunk)
                    temp_zip.flush()
                    
                    # Extract and process ZIP
                    with zipfile.ZipFile(temp_zip.name, 'r') as zip_ref:
                        zip_files = zip_ref.namelist()
                        
                        # Find CSV metadata file
                        csv_file = None
                        for file_name in zip_files:
                            if 'wiserep_spectra' in file_name and file_name.endswith('.csv'):
                                csv_file = file_name
                                break
                        
                        if csv_file:
                            # Read CSV metadata
                            with zip_ref.open(csv_file) as csv_data:
                                csv_content = csv_data.read().decode('utf-8')
                                
                                # Parse CSV
                                csv_reader = csv.DictReader(io.StringIO(csv_content))
                                page_spectra = []
                                
                                for row in csv_reader:
                                    # Extract spectrum information
                                    try:
                                        object_name = row.get('IAU name', row.get('Obj. IAU Name', 'Unknown'))
                                        obs_date = row.get('Obs-date', row.get('Obs-date (UT)', 'Unknown'))
                                        telescope = row.get('Telescope', 'Unknown')
                                        instrument = row.get('Instrument', 'Unknown')
                                        observer = row.get('Observer/s', row.get('Observer', 'Unknown'))
                                        reducer = row.get('Reducer/s', row.get('Reducer', 'Unknown'))
                                        obj_type = row.get('Obj. Type', row.get('Type', 'Unknown'))
                                        spec_quality = row.get('Spec. quality', row.get('Quality', ''))
                                        redshift = row.get('Redshift', '')
                                        ascii_file = row.get('Ascii file', '')
                                        
                                        # Check coordinates if available
                                        try:
                                            spec_ra = float(row.get('Obj. RA', 0))
                                            spec_dec = float(row.get('Obj. DEC', 0))
                                                                                        
                                            # Calculate separation from target
                                            ra_diff = abs(spec_ra - ra) * 3600  # arcseconds
                                            dec_diff = abs(spec_dec - dec) * 3600  # arcseconds
                                            separation = (ra_diff**2 + dec_diff**2)**0.5
                                            
                                            # Only include spectra within reasonable distance (60 arcsec)
                                            if separation > 60:
                                                continue
                                                
                                        except (ValueError, TypeError):
                                            # If coordinate parsing fails, include anyway
                                            separation = 0
                                        
                                        # Try to find corresponding ASCII file
                                        download_link = None
                                        if ascii_file:
                                            # Look for ASCII file in ZIP
                                            for zip_file in zip_files:
                                                if ascii_file in zip_file:
                                                    # For now, provide WiseREP search link
                                                    download_link = f"https://www.wiserep.org/search/spectra?name={object_name}"
                                                    break
                                        
                                        if not download_link:
                                            download_link = f"https://www.wiserep.org/search/spectra?name={object_name}"
                                        
                                        spectrum_info = {
                                            'name': object_name,
                                            'date': obs_date,
                                            'instrument': instrument,
                                            'telescope': telescope,
                                            'observer': observer,
                                            'reducer': reducer,
                                            'spec_type': obj_type,
                                            'quality': spec_quality,
                                            'redshift': redshift,
                                            'download_link': download_link,
                                            'fits_file': ascii_file,
                                            'source': 'WiseREP',
                                            'separation_arcsec': round(separation, 1) if separation > 0 else None
                                        }
                                        
                                        page_spectra.append(spectrum_info)
                                        
                                    except Exception as row_error:
                                        logger.warning(f"Error processing spectrum row: {row_error}")
                                        continue
                                
                                all_spectra.extend(page_spectra)
                                logger.info(f"Page {page_num + 1}: Found {len(page_spectra)} spectra")
                                
                                # If no spectra on this page, we're done
                                if len(page_spectra) == 0:
                                    break
                                    
                        else:
                            logger.warning(f"No CSV metadata file found in page {page_num + 1}")
                            break
                            
            except zipfile.BadZipFile:
                logger.error(f"Invalid ZIP file received from page {page_num + 1}")
                break
            except Exception as zip_error:
                logger.error(f"Error processing ZIP from page {page_num + 1}: {zip_error}")
                break
            
            page_num += 1
        
        # Sort spectra by date (most recent first)
        if all_spectra:
            try:
                all_spectra.sort(key=lambda x: x['date'], reverse=True)
            except:
                pass  # If date sorting fails, keep original order
        
        logger.info(f"Total spectra found for {ztfid}: {len(all_spectra)} across {page_num} pages")
        
        return {
            'ztfid': ztfid,
            'spectra': all_spectra,
            'total_count': len(all_spectra),
            'search_coords': {'ra': ra, 'dec': dec, 'radius_arcsec': search_radius_deg * 3600},
            'source_url': f"https://www.wiserep.org/search/spectra",
            'message': f'Found {len(all_spectra)} spectra within {search_radius_deg * 3600:.0f}" of {ztfid}' if all_spectra else 'No spectra found within search radius'
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching spectra for {ztfid}: {e}")
        return {
            'ztfid': ztfid,
            'spectra': [{
                'name': f'Manual search for {ztfid}',
                'date': 'Network error - try manual search',
                'instrument': 'Various',
                'telescope': 'Various',
                'observer': '',
                'reducer': '',
                'spec_type': 'Check WiseREP',
                'quality': '',
                'redshift': '',
                'download_link': f"https://www.wiserep.org/search/spectra?name={ztfid}",
                'fits_file': '',
                'source': 'WiseREP Manual Search'
            }],
            'total_count': 1,
            'source_url': f"https://www.wiserep.org/search/spectra?name={ztfid}",
            'message': f'Network error - click to search manually',
            'error': 'network_error'
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching spectra for {ztfid}: {e}", exc_info=True)
        return {
            'ztfid': ztfid,
            'spectra': [],
            'total_count': 0,
            'source_url': f"https://www.wiserep.org/search/spectra?name={ztfid}",
            'message': f'Error processing spectra - click to search manually',
            'error': 'processing_error'
        }

@api_app.get("/slack-votes/{ztfid}")
async def get_slack_votes(
    ztfid: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get Slack voting data from GitHub CSV for a ZTFID."""
    import requests
    import pandas as pd
    import io
    
    try:
        # GitHub raw CSV URL - you'll need to update this with your actual URL
        csv_url = "https://raw.githubusercontent.com/YSE-data/YSE-slack-bot/main/vote_data.csv"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(csv_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Read CSV data
        df = pd.read_csv(io.StringIO(response.text))
        
        # Filter for this ZTFID
        object_votes = df[df['ztfid'] == ztfid] if 'ztfid' in df.columns else pd.DataFrame()
        
        votes_data = []
        for _, row in object_votes.iterrows():
            vote_info = {
                'username': row.get('username', 'Unknown'),
                'vote_type': row.get('vote_type', 'unknown'),
                'channel': row.get('channel', 'unknown'),
                'timestamp': row.get('timestamp', 'unknown'),
                'message': f"{row.get('username', 'Unknown')} {row.get('vote_type', 'voted on')} this object in the YSE channel #{row.get('channel', 'unknown')}"
            }
            votes_data.append(vote_info)
        
        logger.info(f"Found {len(votes_data)} Slack votes for {ztfid}")
        return {
            'ztfid': ztfid,
            'votes': votes_data,
            'total_count': len(votes_data),
            'source': 'YSE Slack Bot'
        }
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching Slack votes for {ztfid}: {e}")
        # Return empty data instead of error to not break the page
        return {
            'ztfid': ztfid,
            'votes': [],
            'total_count': 0,
            'source': 'YSE Slack Bot',
            'error': 'Unable to fetch Slack data'
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching Slack votes for {ztfid}: {e}")
        return {
            'ztfid': ztfid,
            'votes': [],
            'total_count': 0,
            'source': 'YSE Slack Bot',
            'error': 'Error processing Slack data'
        }

@api_app.get("/demo/should-show")
async def should_show_demo(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Check if the user should see the demo based on their activity."""
    try:
        # Count user's total votes
        total_votes = db.query(func.count(models.Vote.id)).filter(
            models.Vote.user_id == current_user.id
        ).scalar()
        
        # Show demo if user has fewer than 5 votes
        should_show = total_votes < 5
        
        logger.info(f"Demo check for {current_user.username}: {total_votes} votes, should_show={should_show}")
        
        return {
            'should_show': should_show,
            'total_votes': total_votes,
            'demo_threshold': 5
        }
    except Exception as e:
        logger.error(f"Error checking demo eligibility for user {current_user.username}: {e}")
        # Default to showing demo on error
        return {
            'should_show': True,
            'total_votes': 0,
            'demo_threshold': 5,
            'error': 'Error checking vote count'
        }

@api_app.get("/classifier-badges/{ztfid}")
async def get_classifier_badges(
    ztfid: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get filter badge information for a ZTFID."""
    try:
        badges = filter_manager.get_filter_badge_info(db, ztfid)
        
        return {
            'ztfid': ztfid,
            'badges': badges,
            'total_badges': len(badges)
        }
        
    except Exception as e:
        logger.error(f"Error getting classifier badges for {ztfid}: {e}")
        return {
            'ztfid': ztfid,
            'badges': [],
            'total_badges': 0,
            'error': 'Error loading classifier information'
        }

@api_app.get("/admin/algorithm-config")
async def get_algorithm_config(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the current algorithm configuration (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required",
        )
    
    try:
        if not filter_manager.load_config():
            raise HTTPException(status_code=500, detail="Failed to load configuration")
        
        return filter_manager.config
        
    except Exception as e:
        logger.error(f"Error getting algorithm config: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading configuration: {str(e)}")

@api_app.post("/admin/algorithm-config")
async def save_algorithm_config(
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save the algorithm configuration to YAML file (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required",
        )
    
    try:
        data = await request.json()
        
        # Validate the configuration structure
        required_keys = ['filters', 'settings']
        for key in required_keys:
            if key not in data:
                raise HTTPException(status_code=400, detail=f"Missing required key: {key}")
        
        # Write the configuration to the YAML file
        import yaml
        config_path = filter_manager.config_path
        
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # Reload the configuration in memory
        filter_manager.load_config()
        
        logger.info(f"Algorithm configuration updated by {current_user.username}")
        
        return {
            "status": "success",
            "message": "Algorithm configuration saved successfully",
            "saved_by": current_user.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving algorithm config: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving configuration: {str(e)}")

@api_app.get("/demo/content")
async def get_demo_content(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get demo content with real examples from the upvoted reference bank."""
    try:
        # Get real examples from upvoted reference bank for each science case
        demo_objects = []
        
        science_cases = [
            {
                'case': 'snia-like',
                'description': 'Type Ia supernovae are thermonuclear explosions of white dwarf stars. Notice the smooth light curve and characteristic color evolution from blue to red over ~40 days.',
                'key_features': 'Fast rise (~15-20 days), characteristic color evolution, found in various galaxy types',
                'learning_points': [
                    'Type Ia SNe typically have fast rise times (~15-20 days)',
                    'They show characteristic color evolution from blue to red',
                    'Often found in spiral galaxy arms or elliptical galaxies',
                    'Used as "standard candles" for measuring cosmic distances',
                    'Peak absolute magnitude around -19.3 in V-band'
                ],
                'advanced_features': [
                    'Look for pre-explosion detection in host galaxy',
                    'Check for presence of hydrogen lines (should be absent)',
                    'Monitor for late-time nebular phase (~100+ days)'
                ],
                'science_tags': ['snia-like', 'thermonuclear', 'standard-candle'],
                'photometry_tags': ['fast-rise', 'blue-to-red', 'smooth-decline'],
                'spectra_tags': ['no-hydrogen', 'silicon-lines', 'nebular-phase'],
                'host_tags': ['various-hosts', 'spiral-arm', 'elliptical-ok'],
                'observing_priority': 'high',
                'target_recommendation': 'Excellent target - track color evolution and get spectrum'
            },
            {
                'case': 'ccsn-like', 
                'description': 'Core-collapse supernovae result from massive stars (>8 solar masses) reaching the end of their lives. Note the plateau phase in Type IIP and evidence of hydrogen.',
                'key_features': 'Plateau phase (~100 days), hydrogen signatures, star-forming regions',
                'learning_points': [
                    'Core-collapse SNe come from massive stars (>8 solar masses)',
                    'Type IIP shows a distinctive plateau phase lasting ~100 days',
                    'Often shows hydrogen lines in spectra (Type II)',
                    'Typically found in star-forming regions of spiral galaxies',
                    'More diverse light curve shapes than Type Ia'
                ],
                'advanced_features': [
                    'Monitor for shock breakout in early phases',
                    'Look for circumstellar material interaction',
                    'Track plateau duration and decline rate'
                ],
                'science_tags': ['ccsn-like', 'core-collapse', 'massive-star'],
                'photometry_tags': ['plateau-phase', 'slow-decline', 'red-colors'],
                'spectra_tags': ['hydrogen-lines', 'balmer-series', 'metal-lines'],
                'host_tags': ['spiral-galaxy', 'star-forming', 'hii-regions'],
                'observing_priority': 'high',
                'target_recommendation': 'Priority target - monitor plateau and get early spectrum'
            },
            {
                'case': 'long-lived',
                'description': 'Long-lived transients remain active for months to years, much longer than typical supernovae. These may be tidal disruption events or AGN flares.',
                'key_features': 'Extended duration (months-years), nuclear location, complex light curves',
                'learning_points': [
                    'Long-lived transients can last months to years',
                    'May be tidal disruption events (stars torn apart by black holes)',
                    'Could also be AGN variability or superluminous supernovae',
                    'Often located in galaxy centers or nuclei',
                    'Require long-term monitoring to understand'
                ],
                'advanced_features': [
                    'Check for X-ray and radio counterparts',
                    'Monitor for periodic or quasi-periodic behavior',
                    'Look for broad emission line features'
                ],
                'science_tags': ['long-lived', 'tde-candidate', 'agn-flare'],
                'photometry_tags': ['extended-duration', 'complex-lc', 'multiple-peaks'],
                'spectra_tags': ['broad-lines', 'coronal-lines', 'bowen-fluorescence'],
                'host_tags': ['nuclear-location', 'early-type', 'massive-bh'],
                'observing_priority': 'medium',
                'target_recommendation': 'Long-term monitoring target - needs sustained follow-up'
            },
            {
                'case': 'anomalous',
                'description': 'Anomalous transients have unusual properties that don\'t fit standard classifications. These are exciting because they may represent new physics!',
                'key_features': 'Unusual properties, requires investigation, potential new physics',
                'learning_points': [
                    'Anomalous events don\'t fit standard supernova templates',
                    'May be rare event types like pair-instability SNe or kilonovae',
                    'Could be instrumental artifacts or foreground variables',
                    'Often require detailed follow-up observations',
                    'These discoveries advance our understanding of the universe'
                ],
                'advanced_features': [
                    'Rule out instrumental or calibration issues',
                    'Check for counterparts across electromagnetic spectrum',
                    'Compare with known exotic transient classes'
                ],
                'science_tags': ['anomalous', 'unusual', 'exotic'],
                'photometry_tags': ['atypical-lc', 'unexpected-colors', 'rare-behavior'],
                'spectra_tags': ['unusual-lines', 'unknown-features', 'needs-analysis'],
                'host_tags': ['various-hosts', 'unusual-environment', 'needs-study'],
                'observing_priority': 'urgent',
                'target_recommendation': 'URGENT - Rare discovery! Get all available follow-up'
            }
        ]
        
        # For each science case, find a highly-rated example from the database
        for case_info in science_cases:
            science_case = case_info['case']
            
            # Query for highly upvoted objects in this science case
            example_query = db.query(models.Vote).join(
                models.FeatureBank, models.Vote.ztfid == models.FeatureBank.ztfid
            ).filter(
                models.Vote.science_case == science_case,
                models.Vote.vote_type == 'like'  # Fixed: use vote_type instead of vote
            ).group_by(models.Vote.ztfid).having(
                func.count(models.Vote.id) >= 2  # At least 2 likes
            ).order_by(
                func.count(models.Vote.id).desc(),  # Most liked first
                models.FeatureBank.latest_magnitude.asc()  # Brighter objects preferred
            ).limit(5)  # Get top 5 candidates
            
            example_votes = example_query.all()
            
            if example_votes:
                # Pick the first (most liked) example
                chosen_vote = example_votes[0]
                ztfid = chosen_vote.ztfid
                
                # Get metadata for this object
                feature = db.query(models.FeatureBank).filter(
                    models.FeatureBank.ztfid == ztfid
                ).first()
                
                # Enhanced tag categories for comprehensive demo
                enhanced_tags = {
                    'science': case_info.get('science_tags', ['demo', science_case]),
                    'photometry': case_info.get('photometry_tags', ['well-sampled', 'good-colors']),
                    'spectra': case_info.get('spectra_tags', ['follow-up-needed']),
                    'host': case_info.get('host_tags', ['spiral-galaxy'])
                }
                
                logger.info(f"Selected {ztfid} for {science_case} (coordinates: RA={feature.ra:.4f}, Dec={feature.dec:.4f})")
                
                demo_object = {
                    'ztfid': ztfid,
                    'science_case': science_case,
                    'description': case_info['description'],
                    'enhanced_tags': enhanced_tags,  # New comprehensive tag structure
                    'learning_points': case_info['learning_points'],
                    'key_features': case_info['key_features'],
                    'advanced_features': case_info.get('advanced_features', []),  # New advanced learning
                    'coordinates': {'ra': feature.ra, 'dec': feature.dec} if feature else None,
                    'magnitude': feature.latest_magnitude if feature else None,
                    'vote_count': len(example_votes),
                    'observing_priority': case_info.get('observing_priority', 'medium'),
                    'target_recommendation': case_info.get('target_recommendation', 'Consider for follow-up')
                }
                
                demo_objects.append(demo_object)
                
            else:
                # Fallback: use any object from the feature bank for this science case
                logger.warning(f"No upvoted examples found for {science_case}, using fallback")
                
                fallback_query = db.query(models.FeatureBank).join(
                    models.Vote, models.FeatureBank.ztfid == models.Vote.ztfid
                ).filter(
                    models.Vote.science_case == science_case
                ).order_by(models.FeatureBank.latest_magnitude.asc()).first()
                
                if fallback_query:
                    ztfid = fallback_query.ztfid
                    
                    demo_object = {
                        'ztfid': ztfid,
                        'science_case': science_case,
                        'description': case_info['description'],
                        'tags': ['demo', 'example', science_case.replace('-', '_')],
                        'learning_points': case_info['learning_points'],
                        'key_features': case_info['key_features'],
                        'coordinates': {'ra': fallback_query.ra, 'dec': fallback_query.dec},
                        'magnitude': fallback_query.latest_magnitude,
                        'vote_count': 0
                    }
                    
                    demo_objects.append(demo_object)
                    logger.info(f"Using fallback {ztfid} for {science_case}")
                else:
                    logger.error(f"No examples found for {science_case}, skipping")
        
        if not demo_objects:
            # Ultimate fallback with hardcoded examples
            logger.warning("No database examples found, using hardcoded fallbacks")
            demo_objects = [
                {
                    'ztfid': 'ZTF21aaublej',
                    'science_case': 'snia-like',
                    'description': science_cases[0]['description'],
                    'tags': ['snia-like', 'fast-rise', 'demo'],
                    'learning_points': science_cases[0]['learning_points'],
                    'key_features': science_cases[0]['key_features'],
                    'coordinates': None,
                    'magnitude': None,
                    'vote_count': 0
                }
            ]
        
        logger.info(f"Providing demo with {len(demo_objects)} real examples from reference bank")
        
        return {
            'demo_objects': demo_objects,
            'message': f'Interactive demo with {len(demo_objects)} real examples from upvoted reference bank',
            'source': 'database_examples'
        }
        
    except Exception as e:
        logger.error(f"Error getting demo content for {current_user.username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading demo content: {str(e)}")

@api_app.post("/demo/complete")
async def complete_demo(
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark the demo as completed for the user."""
    try:
        data = await request.json()
        
        # Log the demo completion with details
        demo_completed = data.get('demo_completed', False)
        actions_completed = data.get('actions_completed', {})
        
        logger.info(f"Demo completed by {current_user.username}: {demo_completed}")
        logger.info(f"Actions completed: votes={actions_completed.get('votes', 0)}, tags={actions_completed.get('tags', 0)}, comments={actions_completed.get('comments', 0)}")
        
        # Create a demo "vote" entry to mark completion and prevent demo from showing again
        # We'll use a special demo object identifier
        demo_vote = models.Vote(
            user_id=current_user.id,
            ztfid="DEMO_COMPLETED",
            vote_type="like",  # Fixed: use vote_type instead of vote
            science_case="demo",
            vote_details={
                "demo_completion": True,
                "actions_completed": actions_completed,
                "completion_time": datetime.utcnow().isoformat()
            }
        )
        
        db.add(demo_vote)
        db.commit()
        
        logger.info(f"Demo completion recorded for user {current_user.username}")
        
        return {
            'success': True,
            'message': 'Demo completed successfully!',
            'actions_completed': actions_completed,
            'user_ready': True
        }
        
    except Exception as e:
        logger.error(f"Error completing demo for {current_user.username}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error completing demo: {str(e)}")

# Mount the API app after all routes are defined
app.mount("/api", api_app)

# Background feature extraction function
def run_background_feature_extraction():
    """Run feature extraction in background after startup."""
    try:
        db = next(get_db())
        
        # Check if we can import antares_client before attempting extraction
        try:
            import antares_client
            antares_available = True
        except ImportError:
            antares_available = False
        
        if should_run_feature_extraction(db, max_age_hours=24.0):
            logger.info("Starting automatic feature extraction...")
            try:
                # Use test mode if Antares is not available
                test_mode = not antares_available
                extraction_run = extract_features_for_recent_objects(db, test_mode=test_mode)
                logger.info(f"Automatic feature extraction completed: {extraction_run.objects_processed} objects processed")
            except Exception as e:
                logger.error(f"Automatic feature extraction failed: {e}")
        else:
            if antares_available:
                logger.info("Feature extraction not needed")
            else:
                logger.info("Feature extraction skipped - Antares not available")
        
        db.close()
    except Exception as e:
        logger.error(f"Error in background feature extraction check: {e}", exc_info=True)

# Add startup event - lightweight startup only
@app.on_event("startup")
async def startup_event():
    """Lightweight startup - defer heavy operations to background."""
    logger.info("Application starting up...")
    
    # Load recommender feature bank (quick operation)
    try:
        db = next(get_db())
        recommender_engine.feature_bank = recommender_engine.get_feature_bank_from_db(db)
        logger.info(f"Loaded {len(recommender_engine.feature_bank)} objects into recommender feature bank")
        
        # Process the loaded features (quick operation)
        recommender_engine._load_and_process_features(force_reload=True)
        if recommender_engine.processed_features:
            logger.info(f"Feature processing completed: {len(recommender_engine.processed_features['ztfids'])} objects ready for recommendations")
        else:
            logger.error("Feature processing failed")
        
        db.close()
    except Exception as e:
        logger.error(f"Error loading recommender feature bank: {e}", exc_info=True)
    
    # Schedule feature extraction as background task instead of blocking startup
    try:
        import threading
        # Run feature extraction in background thread after 5 second delay
        def delayed_extraction():
            import time
            time.sleep(5)  # Allow server to fully start
            run_background_feature_extraction()
        
        bg_thread = threading.Thread(target=delayed_extraction, daemon=True)
        bg_thread.start()
    except Exception as e:
        logger.error(f"Error scheduling background feature extraction: {e}", exc_info=True)

# Exception Handler for HTTP Exceptions (including CSRF and 404)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 401:
        # Check if this is an HTML request (not an API call)
        accept_header = request.headers.get("accept", "")
        is_html_request = "text/html" in accept_header or request.url.path.startswith("/api/") == False
        
        logger.warning(f"401 Unauthorized for {request.method} {request.url.path} - HTML request: {is_html_request}")
        
        if is_html_request and not request.url.path.startswith("/api/"):
            # Redirect to login page for HTML requests
            logger.info(f"Redirecting {request.url.path} to login page due to 401")
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/login", status_code=302)
        else:
            # Return JSON error for API requests
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"}
            )
    elif exc.status_code == 403:  # CSRF error
        logger.warning(f"CSRF protection error: {exc.detail}", exc_info=True)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail or "CSRF token verification failed"}
        )
    elif exc.status_code == 404:  # Not Found
        return JSONResponse(
            status_code=404,
            content={"detail": "Not Found"}
        )
    # For other HTTP exceptions, return a JSON response with the status and detail
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail or str(exc)}
    )

# General Exception Handler (for unhandled exceptions)
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    # In a real production environment, you might want to send this to an error tracking service
    # For now, returning a generic 500 error
    return PlainTextResponse("An internal server error occurred.", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888) 