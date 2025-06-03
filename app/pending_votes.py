"""
Pending vote management for automated classifiers.

This module provides functions for automated classifiers to add objects
to science case pending queues, which are then prioritized in recommendations.
"""

import logging
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from . import models

logger = logging.getLogger(__name__)


def create_pending_vote_for_science_case(db: Session, ztfid: str, science_case: str, details: Optional[Dict] = None):
    """
    Create a pending vote for any science case.
    
    This allows automated classifiers to add objects to specific science case queues.
    
    Args:
        db: Database session
        ztfid: ZTF ID of the object
        science_case: Science case (e.g., "anomalous", "snia-like", "ccsn-like", etc.)
        details: Optional dictionary with additional details about why this object was flagged
    """
    try:
        # Check if a pending vote already exists for this object and science case
        existing_vote = db.query(models.Vote).filter(
            models.Vote.ztfid == ztfid,
            models.Vote.vote_type == "pending",
            models.Vote.science_case == science_case
        ).first()
        
        if not existing_vote:
            # Prepare vote details
            vote_details = {"auto_generated": True}
            if details:
                vote_details.update(details)
            
            # Create a system-generated pending vote
            pending_vote = models.Vote(
                user_id=None,  # System-generated, no specific user
                ztfid=ztfid,
                vote_type="pending",
                science_case=science_case,
                vote_details=vote_details
            )
            db.add(pending_vote)
            logger.info(f"Created pending vote for {science_case} science case: {ztfid}")
        else:
            logger.debug(f"Pending vote already exists for {ztfid} in {science_case} science case")
    
    except Exception as e:
        logger.error(f"Error creating pending vote for {ztfid} in {science_case}: {e}")


def get_pending_objects_for_science_case(db: Session, science_case: str) -> List[str]:
    """
    Get list of objects that are pending for a specific science case.
    
    Args:
        db: Database session
        science_case: Science case to get pending objects for
        
    Returns:
        List of ZTFIDs that are pending for the science case
    """
    pending_votes = db.query(models.Vote).filter(
        models.Vote.vote_type == "pending",
        models.Vote.science_case == science_case
    ).all()
    
    return [vote.ztfid for vote in pending_votes]


def remove_pending_vote(db: Session, ztfid: str, science_case: str):
    """
    Remove a pending vote for an object from a specific science case.
    
    This is called when a user votes on a pending object to remove it from the pending queue.
    
    Args:
        db: Database session
        ztfid: ZTF ID of the object
        science_case: Science case to remove pending vote from
    """
    try:
        pending_vote = db.query(models.Vote).filter(
            models.Vote.ztfid == ztfid,
            models.Vote.vote_type == "pending",
            models.Vote.science_case == science_case
        ).first()
        
        if pending_vote:
            db.delete(pending_vote)
            logger.info(f"Removed pending vote for {ztfid} from {science_case} science case")
        
    except Exception as e:
        logger.error(f"Error removing pending vote for {ztfid} from {science_case}: {e}") 