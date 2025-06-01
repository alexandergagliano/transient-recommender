"""Database models for the transient recommender server."""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Table, UniqueConstraint, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base

class User(Base):
    """User model for authentication."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    api_key = Column(String, unique=True, nullable=True)
    data_sharing_consent = Column(Boolean, default=False)
    science_interests = Column(JSON, nullable=True)  # Added field for science interests
    
    # Relationships
    votes = relationship("Vote", back_populates="user")
    tags = relationship("Tag", back_populates="user")
    notes = relationship("Note", back_populates="user")
    comments = relationship("Comment", back_populates="user")
    audio_notes = relationship("AudioNote", back_populates="user")
    password_reset_tokens = relationship("PasswordResetToken", back_populates="user")

class Vote(Base):
    """Vote model for storing user preferences."""
    
    __tablename__ = "votes"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    ztfid = Column(String, index=True)
    vote_type = Column(String)  # "like", "dislike", "target", or "skip"
    science_case = Column(String)
    vote_details = Column(JSON, nullable=True)  # Renamed from metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="votes")
    
    # Modified constraint: prevent duplicate votes of the same type, but allow different vote types
    __table_args__ = (
        UniqueConstraint('user_id', 'ztfid', 'vote_type', name='unique_user_vote_type'),
    )

class Tag(Base):
    """Tag model for storing object classifications."""
    
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    ztfid = Column(String, index=True)
    tag_name = Column(String, index=True)
    category = Column(String, default="general")  # science, spectra, photometry, host, general
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="tags")
    
    # Add a unique constraint to prevent duplicate tags
    __table_args__ = (
        UniqueConstraint('user_id', 'ztfid', 'tag_name', 'category', name='unique_user_tag_category'),
    )

class Note(Base):
    """Note model for storing user notes on objects."""
    
    __tablename__ = "notes"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    ztfid = Column(String, index=True)
    text = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="notes")
    
    # Add a unique constraint to have one note per user per object
    __table_args__ = (
        UniqueConstraint('user_id', 'ztfid', name='unique_user_note'),
    )

class Comment(Base):
    """Comment model for storing individual user comments on objects."""
    
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    ztfid = Column(String, index=True)
    text = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="comments")

class FeatureBank(Base):
    """Feature bank model for storing object features."""
    
    __tablename__ = "feature_bank"
    
    id = Column(Integer, primary_key=True, index=True)
    ztfid = Column(String, unique=True, index=True)
    ra = Column(Float)
    dec = Column(Float)
    latest_magnitude = Column(Float, nullable=True)
    features = Column(JSON)  # Store computed features as JSON
    feature_errors = Column(JSON)  # Store feature errors as JSON
    processed_features = Column(JSON, nullable=True)  # Store preprocessed/scaled features
    additional_details = Column(JSON, nullable=True)  # Renamed from metadata
    mjd_extracted = Column(Float)  # MJD when features were extracted
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# New Model for Audio Notes
class AudioNote(Base):
    __tablename__ = "audio_notes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    ztfid = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String, nullable=True) # Can be None if not available
    audio_data = Column(LargeBinary) # Storing audio directly for now
    filename = Column(String, nullable=True) # Optional: if we want to give it a name
    content_type = Column(String, nullable=True) # e.g., 'audio/webm' or 'audio/mp3'
    transcription = Column(String, nullable=True) # For speech-to-text

    user = relationship("User", back_populates="audio_notes")

    __table_args__ = (
        UniqueConstraint('user_id', 'ztfid', 'timestamp', name='unique_user_audio_note'), # Ensure unique entry per user, object, and time
    )

class FeatureExtractionRun(Base):
    """Model to track feature extraction runs."""
    
    __tablename__ = "feature_extraction_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_date = Column(DateTime, default=datetime.utcnow)
    mjd_run = Column(Float)  # MJD when extraction was run
    lookback_days = Column(Float, default=20.0)  # How many days back we looked
    objects_found = Column(Integer, default=0)  # Number of objects found
    objects_processed = Column(Integer, default=0)  # Number of objects successfully processed
    status = Column(String, default="running")  # "running", "completed", "failed"
    error_message = Column(String, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class PasswordResetToken(Base):
    """Password reset token model for secure password resets."""
    
    __tablename__ = "password_reset_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    used = Column(Boolean, default=False)
    used_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="password_reset_tokens") 