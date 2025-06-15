"""Web-based transient recommender engine."""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc # Added for ordering by timestamp
from datetime import datetime, timedelta
from astropy.time import Time
import json
import logging
import time
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os

# Astroplan imports
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astroplan import Observer, FixedTarget, observability_table, is_observable, AltitudeConstraint, AirmassConstraint, AtNightConstraint
from astroplan.constraints import TimeConstraint

from . import models
from .pending_votes import get_pending_objects_for_science_case

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Science Case Definitions (from orig_recommender.py)
SCIENCE_SEEDS = {
    "long-lived": ['ZTF23abaeenk','ZTF21abulpnc'],
    "anomalous":['ZTF22abcvfgs','ZTF21abgjldn','ZTF23aaawbsc'],
    "snia-like": ['ZTF21aaublej', 'ZTF20abucvpo', 'ZTF20acrudzk', 'ZTF22aaptcfi'],
    "ccsn-like": ['ZTF20aclnqle', 'ZTF19adaivcf', 'ZTF22aaohilj', 'ZTF22abynkpz'],
    "precursor": ['ZTF24abymeet'],
    "all": [] # 'all' means no specific upweighting/downweighting based on pre-defined seeds initially
}

SCIENCE_DOWNVOTES = {
    "long-lived": SCIENCE_SEEDS['snia-like'],
    "anomalous": [],
    "precursor": SCIENCE_SEEDS['snia-like'],
    "snia-like": list(np.concatenate([SCIENCE_SEEDS['ccsn-like'], SCIENCE_SEEDS['long-lived']])),
    "ccsn-like": SCIENCE_SEEDS['snia-like'],
    "all": []
}

# Feature selection
BASE_FEATURES = [
    'g_peak_mag', 'r_peak_mag', 'g_peak_time', 'g_rise_time', 'g_decline_time',
    'g_duration_above_half_flux', 'r_duration_above_half_flux',
    'r_peak_time', 'r_rise_time', 'r_decline_time',
    'mean_g-r', 'g-r_at_g_peak', 'mean_color_rate',
    'g_mean_rolling_variance', 'r_mean_rolling_variance',
    'g_rise_local_curvature', 'g_decline_local_curvature',
    'r_rise_local_curvature', 'r_decline_local_curvature'
]

EARLY_FEATURES = [
    'g_rise_time', 'g_duration_above_half_flux',
    'r_rise_time', 'mean_g-r', 'mean_color_rate',
    'g_mean_rolling_variance', 'r_mean_rolling_variance',
    'g_rise_local_curvature', 'r_rise_local_curvature'
]

class NoObjectsAvailableError(Exception):
    """Raised when no objects match the specified constraints."""
    def __init__(self, message: str, active_constraints: dict):
        super().__init__(message)
        self.active_constraints = active_constraints

class WebRecommender:
    """Web-based transient recommender."""
    
    def __init__(self, feature_bank_path: Optional[str] = None):
        """Initialize the recommender with an optional path to a feature bank CSV."""
        self.feature_bank_path = feature_bank_path
        self.feature_bank = None
        self.processed_features = None
        self.imputer = None
        self.scaler = None
        self.last_load_time = None
        
        # Performance settings
        self.max_sample_size = 5000  # Limit feature bank size for performance
        self.cache_timeout = 3600  # Cache for 1 hour
        
        logger.info(f"WebRecommender initialized with max_sample_size={self.max_sample_size}")
    
    def _load_and_process_features(self, force_reload=False):
        """Load and process feature bank with imputation and scaling."""
        current_time = time.time()
        
        # Use cached features if available and not expired
        if (not force_reload and 
            self.processed_features is not None and 
            self.last_load_time is not None and 
            (current_time - self.last_load_time) < self.cache_timeout):
            logger.info("Using cached processed features")
            return
        
        logger.info("Loading and processing feature bank...")
        start_time = time.time()
        
        if self.feature_bank is None or len(self.feature_bank) == 0:
            logger.warning("No feature bank available")
            return
        
        logger.info(f"Processing features for {len(self.feature_bank)} objects")
        
        # Sample if too large for performance and store the sampled version
        feature_bank_to_process = self.feature_bank
        if len(self.feature_bank) > self.max_sample_size:
            logger.info(f"Sampling {self.max_sample_size} objects from {len(self.feature_bank)} for performance")
            if not hasattr(self, 'sampled_feature_bank') or self.sampled_feature_bank is None:
                self.sampled_feature_bank = self.feature_bank.sample(n=self.max_sample_size, random_state=42)
            feature_bank_to_process = self.sampled_feature_bank
        elif not hasattr(self, 'sampled_feature_bank') or self.sampled_feature_bank is None:
            self.sampled_feature_bank = self.feature_bank
        
        # Check which BASE_FEATURES are available in the DataFrame
        available_features = [f for f in BASE_FEATURES if f in feature_bank_to_process.columns]
        missing_features = [f for f in BASE_FEATURES if f not in feature_bank_to_process.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        if not available_features:
            logger.error("No BASE_FEATURES found in DataFrame")
            return
        
        logger.info(f"Using {len(available_features)}/{len(BASE_FEATURES)} available features")
        
        try:
            # Extract feature data directly from DataFrame columns
            X = feature_bank_to_process[available_features].values
            ztfids = feature_bank_to_process['ZTFID'].values
            
            # For errors, use a simple default (10% of feature value, min 0.01)
            X_err = np.abs(X) * 0.1
            X_err = np.clip(X_err, 0.01, None)
            
            logger.info(f"Extracted features shape: {X.shape}")
            logger.info(f"Features with valid data: {np.sum(~np.isnan(X).all(axis=1))} / {len(X)}")
            
            # Fast imputation with fewer neighbors for speed
            logger.info("Performing feature imputation...")
            impute_start = time.time()
            self.imputer = KNNImputer(n_neighbors=min(3, len(X)-1), weights='distance')
            X_filled = self.imputer.fit_transform(X)
            X_err_filled = self.imputer.transform(X_err)
            logger.info(f"Imputation took {time.time() - impute_start:.2f} seconds")
            
            # Scale features
            logger.info("Scaling features...")
            scale_start = time.time()
            self.scaler = StandardScaler()
            X_filled_scaled = self.scaler.fit_transform(X_filled)
            X_err_filled_scaled = X_err_filled / self.scaler.scale_
            
            # Clip error floor
            error_floor = 1.e-3
            X_err_filled_scaled = np.clip(X_err_filled_scaled, error_floor, None)
            logger.info(f"Scaling took {time.time() - scale_start:.2f} seconds")
            
            # Store processed features
            self.processed_features = {
                'X_scaled': X_filled_scaled,
                'X_err_scaled': X_err_filled_scaled,
                'feature_cols': available_features,
                'feature_cols_err': [f"{col}_err" for col in available_features],
                'ztfids': ztfids
            }
            
            self.last_load_time = current_time
            total_time = time.time() - start_time
            logger.info(f"Feature processing completed in {total_time:.2f} seconds")
            logger.info(f"Processed {len(ztfids)} objects with {len(available_features)} features each")
            
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            logger.error(f"Available features: {available_features}")
            logger.error(f"DataFrame shape: {feature_bank_to_process.shape}")
            if len(available_features) > 0:
                logger.error(f"Sample feature data: {feature_bank_to_process[available_features].iloc[:2].values}")
            self.processed_features = None
    
    def pairwise_chisq_distances(self, query, query_err, X, X_err):
        """Calculate chi-square distances between query and all points in X."""
        diff = X - query
        total_err = X_err**2 + query_err**2
        # Add small constant to avoid division by zero
        total_err = np.maximum(total_err, 1e-10)
        chisq = np.sum(diff**2 / total_err, axis=1)
        return chisq
    
    def apply_feedback_reweighting(self, base_scores, feature_matrix, liked_ztfids, disliked_ztfids, 
                                   ztfids, alpha=3.0, beta=3.0):
        """Apply feedback reweighting based on liked and disliked objects."""
        final_scores = base_scores.copy()
        liked_indices = [i for i, ztfid in enumerate(ztfids) if ztfid in liked_ztfids]
        disliked_indices = [i for i, ztfid in enumerate(ztfids) if ztfid in disliked_ztfids]
        
        for i in range(len(base_scores)):
            penalties = [np.exp(-alpha * np.linalg.norm(feature_matrix[i] - feature_matrix[j])) 
                        for j in disliked_indices]
            boosts = [np.exp(-beta * np.linalg.norm(feature_matrix[i] - feature_matrix[j])) 
                     for j in liked_indices]
            final_scores[i] -= sum(penalties)
            final_scores[i] += sum(boosts)
        return final_scores
    
    def get_feature_bank_from_db(self, db: Session) -> pd.DataFrame:
        """Get feature bank from database, falling back to CSV if needed."""
        logger.info("Loading feature bank from database...")
        
        try:
            # Get all features from database
            db_features = db.query(models.FeatureBank).all()
            logger.info(f"Found {len(db_features)} objects in database")
            
            if not db_features:
                logger.warning("No features found in database, falling back to CSV")
                return self.get_feature_bank_from_csv()
            
            # Convert to DataFrame
            feature_data = []
            for obj in db_features:
                try:
                    # Combine basic info with features and errors
                    row_data = {
                        'ZTFID': obj.ztfid,
                        'ra': obj.ra,
                        'dec': obj.dec,
                        'latest_magnitude': obj.latest_magnitude,
                        'mjd_extracted': obj.mjd_extracted
                    }
                    
                    # Add features
                    if obj.features:
                        row_data.update(obj.features)
                    
                    # Add feature errors
                    if obj.feature_errors:
                        row_data.update(obj.feature_errors)
                    
                    feature_data.append(row_data)
                except Exception as e:
                    logger.error(f"Error processing object {obj.ztfid}: {e}")
                    continue
            
            df = pd.DataFrame(feature_data)
            logger.info(f"Created DataFrame with {len(df)} objects and columns: {list(df.columns)}")
            
            # Check for required columns
            required_cols = ['ZTFID', 'ra', 'dec']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()
            
            # If database has fewer than 1000 objects, supplement with CSV
            if len(df) < 1000:
                logger.info("Database has few objects, supplementing with CSV data...")
                csv_df = self.get_feature_bank_from_csv()
                
                # Only add CSV objects not already in database
                csv_ztfids = set(csv_df['ZTFID'])
                db_ztfids = set(df['ZTFID'])
                new_ztfids = csv_ztfids - db_ztfids
                
                if new_ztfids:
                    csv_supplement = csv_df[csv_df['ZTFID'].isin(new_ztfids)]
                    df = pd.concat([df, csv_supplement], ignore_index=True)
                    logger.info(f"Added {len(csv_supplement)} objects from CSV, total: {len(df)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading feature bank from database: {e}", exc_info=True)
            logger.warning("Falling back to CSV")
            return self.get_feature_bank_from_csv()

    def get_feature_bank_from_csv(self) -> pd.DataFrame:
        """Get feature bank from CSV file (fallback method)."""
        logger.info(f"Loading feature bank from {self.feature_bank_path}")
        
        if not os.path.exists(self.feature_bank_path):
            logger.error(f"Feature bank file not found: {self.feature_bank_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.feature_bank_path)
        logger.info(f"Loaded {len(df)} objects from CSV")
        return df
    
    def update_feature_bank(self, db: Session):
        """Update the feature bank from the CSV file."""
        if not self.feature_bank_path:
            logger.warning("No feature bank path provided, cannot update")
            return
        
        try:
            new_feature_bank = pd.read_csv(self.feature_bank_path)
            self.feature_bank = new_feature_bank
            
            for _, row in new_feature_bank.iterrows():
                ztfid = row['ZTFID']
                
                # Check if object already exists
                obj = db.query(models.FeatureBank).filter(models.FeatureBank.ztfid == ztfid).first()
                
                # Prepare features and errors dictionaries
                feature_cols = [col for col in row.index if col not in ['ZTFID', 'ra', 'dec', 'latest_magnitude', 'mjd_extracted'] 
                               and not col.endswith('_err')]
                error_cols = [col for col in row.index if col.endswith('_err')]
                
                features = {col: float(row[col]) if pd.notna(row[col]) and np.isfinite(row[col]) else None for col in feature_cols}
                errors = {col.replace('_err', ''): float(row[col]) if pd.notna(row[col]) and np.isfinite(row[col]) else None for col in error_cols}
                
                if obj:
                    # Update existing object
                    obj.ra = float(row['ra']) if pd.notna(row['ra']) and np.isfinite(row['ra']) else None
                    obj.dec = float(row['dec']) if pd.notna(row['dec']) and np.isfinite(row['dec']) else None
                    obj.latest_magnitude = float(row.get('latest_magnitude')) if pd.notna(row.get('latest_magnitude')) and np.isfinite(row.get('latest_magnitude')) else None
                    obj.features = features
                    obj.feature_errors = errors
                    obj.mjd_extracted = float(row['mjd_extracted']) if pd.notna(row['mjd_extracted']) and np.isfinite(row['mjd_extracted']) else None
                    obj.last_updated = datetime.utcnow()
                else:
                    # Create new object
                    new_obj = models.FeatureBank(
                        ztfid=ztfid,
                        ra=float(row['ra']) if pd.notna(row['ra']) and np.isfinite(row['ra']) else None,
                        dec=float(row['dec']) if pd.notna(row['dec']) and np.isfinite(row['dec']) else None,
                        latest_magnitude=float(row.get('latest_magnitude')) if pd.notna(row.get('latest_magnitude')) and np.isfinite(row.get('latest_magnitude')) else None,
                        features=features,
                        feature_errors=errors,
                        mjd_extracted=float(row['mjd_extracted']) if pd.notna(row['mjd_extracted']) and np.isfinite(row['mjd_extracted']) else None
                    )
                    db.add(new_obj)
            
            db.commit()
            logger.info(f"Updated feature bank with {len(new_feature_bank)} objects")
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating feature bank: {e}")
    
    def get_user_votes(self, db: Session, user_id: int) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Get the user's votes."""
        votes = db.query(models.Vote).filter(models.Vote.user_id == user_id).all()
        
        liked = [vote.ztfid for vote in votes if vote.vote_type == "like"]
        disliked = [vote.ztfid for vote in votes if vote.vote_type == "dislike"]
        targets = [vote.ztfid for vote in votes if vote.vote_type == "target"] # Keep targets separate
        skipped = [vote.ztfid for vote in votes if vote.vote_type == "skip"]
        
        return liked, disliked, targets, skipped
    
    def get_user_positive_preference_history(self, db: Session, user_id: int) -> List[models.Vote]:
        """Get user's like and target votes, ordered by most recent."""
        return db.query(models.Vote).filter(
            models.Vote.user_id == user_id,
            models.Vote.vote_type.in_(["like", "target"])
        ).order_by(desc(models.Vote.last_updated)).all()
    
    def generate_recommendation_explanation(self, db: Session, ztfid: str, user_id: int, science_case: str) -> str:
        """
        Generate a personalized explanation for why an object was recommended.
        
        Args:
            db: Database session
            ztfid: ZTF ID of the recommended object
            user_id: User ID to personalize the explanation
            science_case: Current science case being recommended for
            
        Returns:
            A string explaining why this object was recommended
        """
        try:
            # Get the recommended object's features
            obj_idx = np.where(self.processed_features['ztfids'] == ztfid)[0]
            if len(obj_idx) == 0:
                return "Recommended based on its features"
            
            obj_features = self.processed_features['X_scaled'][obj_idx[0]]
            
            # Get user's liked objects
            liked_votes = db.query(models.Vote).filter(
                models.Vote.user_id == user_id,
                models.Vote.vote_type.in_(["like", "target"])
            ).all()
            
            # Get other users' likes for this object in the same science case
            other_users_likes = db.query(models.Vote).filter(
                models.Vote.ztfid == ztfid,
                models.Vote.user_id != user_id,  # Exclude current user
                models.Vote.vote_type.in_(["like", "target"]),
                models.Vote.science_case == science_case
            ).all()
            
            if not liked_votes:
                if other_users_likes:
                    # Other users liked this object in the same science case
                    return f"Recommended because it was liked by others interested in {science_case} objects"
                
                # No user history - use science case seeds
                for seed_ztf in SCIENCE_SEEDS.get(science_case, []):
                    seed_idx = np.where(self.processed_features['ztfids'] == seed_ztf)[0]
                    if len(seed_idx) > 0:
                        return f"Recommended because it's similar to known {science_case} objects"
                return f"Recommended for {science_case} science case"
            
            # Find nearest liked object
            min_dist = float('inf')
            nearest_ztfid = None
            nearest_science_case = None
            
            for vote in liked_votes:
                vote_idx = np.where(self.processed_features['ztfids'] == vote.ztfid)[0]
                if len(vote_idx) > 0:
                    vote_features = self.processed_features['X_scaled'][vote_idx[0]]
                    dist = np.linalg.norm(obj_features - vote_features)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_ztfid = vote.ztfid
                        nearest_science_case = vote.science_case
            
            if nearest_ztfid:
                if nearest_science_case == science_case:
                    return f"Recommended because you liked {nearest_ztfid}"
                else:
                    return f"Recommended because you liked {nearest_ztfid} (a {nearest_science_case} object)"
            
            # If no personal likes match but others liked this object
            if other_users_likes:
                return f"Recommended because it was liked by others interested in {science_case} objects"
            
            return f"Recommended for {science_case} science case"
            
        except Exception as e:
            logger.error(f"Error generating recommendation explanation: {e}")
            return f"Recommended for {science_case} science case"
    
    def get_recommendations(self, db: Session, user_id: int, k: int = 10, science_case: str = "snia-like",
                            obs_telescope: Optional[str] = None, 
                            obs_days: Optional[int] = None, 
                            obs_mag_limit: Optional[float] = None,
                            start_ztfid: Optional[str] = None,
                            realtime_mode: bool = False,
                            recent_days: int = 7) -> List[Dict]:
        """Get recommendations for a user."""
        logger.info(f"Getting recommendations for user {user_id}, science_case: {science_case}")
        start_time = time.time()
        
        try:
            # Get feature bank
            if self.feature_bank is None:
                logger.info("Feature bank is None, loading from database...")
                self.feature_bank = self.get_feature_bank_from_db(db)
                
                if len(self.feature_bank) == 0:
                    logger.error("No feature bank available")
                    return []
                
                logger.info(f"Loaded feature bank with {len(self.feature_bank)} objects")
                
                # Sample for performance if needed and store the sampled version
                if len(self.feature_bank) > self.max_sample_size:
                    logger.info(f"Sampling {self.max_sample_size} objects from {len(self.feature_bank)} for performance")
                    self.sampled_feature_bank = self.feature_bank.sample(n=self.max_sample_size, random_state=42)
                else:
                    logger.info("Using full feature bank (no sampling needed)")
                    self.sampled_feature_bank = self.feature_bank
                
                logger.info(f"Sampled feature bank has {len(self.sampled_feature_bank)} objects")
                
                # Process features if not already done
                if not hasattr(self, 'processed_features') or self.processed_features is None:
                    logger.info("Performing feature imputation...")
                    start_time = time.time()
                    self._load_and_process_features()
                    logger.info(f"Feature processing completed in {time.time() - start_time:.2f} seconds")
                else:
                    logger.info("Using cached processed features")
            
            if self.processed_features is None:
                logger.error("No processed features available")
                return []
            
            if not hasattr(self, 'sampled_feature_bank') or self.sampled_feature_bank is None:
                logger.error("sampled_feature_bank is not initialized")
                return []
            
            logger.info(f"Feature bank status - feature_bank: {self.feature_bank is not None}, sampled_feature_bank: {self.sampled_feature_bank is not None}, processed_features: {self.processed_features is not None}")
            
            # Get user votes
            liked, disliked, targets, skipped = self.get_user_votes(db, user_id)
            excluded_ids = set(liked + disliked + skipped)
            
            logger.info(f"User has voted on {len(excluded_ids)} objects: {len(liked)} liked, {len(disliked)} disliked, {len(skipped)} skipped")
            if start_ztfid:
                logger.info(f"start_ztfid {start_ztfid} in excluded_ids: {start_ztfid in excluded_ids}")
            
            # Get processed features
            X_scaled = self.processed_features['X_scaled']
            X_err_scaled = self.processed_features['X_err_scaled']
            ztfids = self.processed_features['ztfids']
            feature_cols = self.processed_features['feature_cols']
            feature_cols_err = self.processed_features['feature_cols_err']
            
            # Create mask for available objects (not voted on)
            available_mask = ~np.isin(ztfids, list(excluded_ids))
            logger.info(f"After excluding voted objects: {np.sum(available_mask)} objects available")
            
            # Get available objects
            X_available = X_scaled[available_mask]
            X_err_available = X_err_scaled[available_mask]
            ztfids_available = ztfids[available_mask]
            
            if len(X_available) == 0:
                logger.error("No available objects after filtering")
                return []
            
            logger.info(f"Found {len(X_available)} available objects for recommendations")
            
            # Initialize query vector
            query_vector = None
            query_vector_err = None
            query_ztfid = None
            
            # Get pending objects for this science case
            pending_science_objects = []
            if science_case and science_case != "all":
                pending_science_objects = get_pending_objects_for_science_case(db, science_case)
                logger.info(f"Found {len(pending_science_objects)} pending objects for {science_case}")
            
            # Rest of the method remains the same...
            # ... (rest of the method content remains unchanged)
        except Exception as e:
            logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
            raise 