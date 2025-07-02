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
    vote_type = Column(String)  # "like", "dislike", "target", "skip", or "pending"
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
    last_detection_mjd = Column(Float, nullable=True)  # MJD of last detection for real-time filtering
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
    is_automatic = Column(Boolean, default=True)  # True for automatic runs, False for manual

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

class AnomalyDetectionResult(Base):
    """Model to store anomaly detection results."""
    
    __tablename__ = "anomaly_detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    ztfid = Column(String, index=True)
    anomaly_score = Column(Float)  # Overall anomaly score (percentage)
    mjd_anom = Column(JSON, nullable=True)  # MJD values where anomalies were detected
    anom_scores = Column(JSON, nullable=True)  # Individual anomaly scores at each MJD
    norm_scores = Column(JSON, nullable=True)  # Normalization scores
    detection_threshold = Column(Float, default=60.0)  # Threshold used for detection
    is_anomalous = Column(Boolean, default=False)  # Whether this object exceeded threshold
    feature_extraction_run_id = Column(Integer, ForeignKey("feature_extraction_runs.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Add unique constraint to prevent duplicate results for same object
    __table_args__ = (
        UniqueConstraint('ztfid', name='unique_anomaly_result'),
    )

class AnomalyNotification(Base):
    """Model to track anomaly detection notifications."""
    
    __tablename__ = "anomaly_notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    detection_run_id = Column(Integer, ForeignKey("feature_extraction_runs.id"))
    objects_detected = Column(Integer, default=0)  # Number of anomalous objects found
    ztfids_detected = Column(JSON, nullable=True)  # List of ZTFIDs that were flagged
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False)  # Whether user has seen this notification
    acknowledged_at = Column(DateTime, nullable=True)

class PendingVote(Base):
    """Model to track pending votes from automated classifiers."""
    
    __tablename__ = "pending_votes"
    
    id = Column(Integer, primary_key=True, index=True)
    ztfid = Column(String, index=True)
    science_case = Column(String, index=True)  # "snia-like", "ccsn-like", etc.
    confidence = Column(Float)  # Confidence score from classifier
    classifier_name = Column(String, nullable=True)  # Name of the classifier that made this prediction
    additional_info = Column(JSON, nullable=True)  # Any additional metadata from classifier
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)  # Whether this has been acted upon
    resolved_at = Column(DateTime, nullable=True)
    resolution_type = Column(String, nullable=True)  # "accepted", "rejected", "skipped"
    
    # Add unique constraint to prevent duplicate pending votes
    __table_args__ = (
        UniqueConstraint('ztfid', 'science_case', 'classifier_name', name='unique_pending_vote'),
    )

class ClassifierApprovalRequest(Base):
    """Model to track approval requests for custom classifiers."""
    
    __tablename__ = "classifier_approval_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    classifier_name = Column(String, index=True)
    science_case = Column(String, index=True)
    python_code = Column(String)  # The actual Python code to be executed
    code_hash = Column(String, unique=True, index=True)  # SHA256 hash of the code for tracking
    confidence_threshold = Column(Float, default=50.0)
    description = Column(String, nullable=True)
    url = Column(String, nullable=True)  # Documentation URL
    
    # Request tracking
    requesting_user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="pending")  # "pending", "approved", "rejected"
    security_validated = Column(Boolean, default=False)
    
    # Review tracking
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    rejection_reason = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    requesting_user = relationship("User", foreign_keys=[requesting_user_id])
    reviewing_user = relationship("User", foreign_keys=[reviewed_by])

class SecurityAuditLog(Base):
    """Model to log security-related events for audit purposes."""
    
    __tablename__ = "security_audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True)  # "classifier_security_violation", "unauthorized_access", etc.
    classifier_name = Column(String, nullable=True, index=True)
    ztfid = Column(String, nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    ip_address = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    additional_data = Column(JSON, nullable=True)
    severity = Column(String, default="MEDIUM")  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])

class CustomClassifierResult(Base):
    """Model to store results from custom classifiers."""
    
    __tablename__ = "custom_classifier_results"
    
    id = Column(Integer, primary_key=True, index=True)
    ztfid = Column(String, index=True)
    classifier_name = Column(String, index=True)
    science_case = Column(String, index=True)
    classification_label = Column(String)  # The label returned by the classifier
    confidence_score = Column(Float)  # The score returned by the classifier
    execution_time = Column(Float, nullable=True)  # How long the classifier took to run
    
    # Approval tracking
    approval_request_id = Column(Integer, ForeignKey("classifier_approval_requests.id"), nullable=True)
    
    # Metadata
    additional_info = Column(JSON, nullable=True)
    feature_extraction_run_id = Column(Integer, ForeignKey("feature_extraction_runs.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    approval_request = relationship("ClassifierApprovalRequest")
    
    # Unique constraint to prevent duplicate results
    __table_args__ = (
        UniqueConstraint('ztfid', 'classifier_name', 'science_case', name='unique_classifier_result'),
    ) 