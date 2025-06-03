"""Anomaly detection service for transient objects."""

import sys
import os
import logging
import time
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from datetime import datetime
import json

from . import models
from .pending_votes import create_pending_vote_for_science_case, get_pending_objects_for_science_case

logger = logging.getLogger(__name__)

class AnomalyDetectionService:
    """Service for running anomaly detection on transient objects."""
    
    def __init__(self):
        """Initialize the anomaly detection service."""
        self.client = None
        self.initialized = False
        self.anomaly_threshold = 60.0  # Default threshold (60%)
        
        # Feature configuration matching the example
        self.lc_features = [
            'mean_g-r', 'g_peak_mag', 'r_peak_mag', 'g_amplitude', 'r_amplitude', 
            'g-r_at_g_peak', 'mean_color_rate', 'g_peak_time', 'r_peak_time', 
            'g_rise_time', 'r_rise_time', 'g_decline_time', 'r_decline_time', 
            'g_duration_above_half_flux', 'r_duration_above_half_flux', 
            'g_beyond_2sigma', 'r_beyond_2sigma', 'g_mean_rolling_variance', 
            'r_mean_rolling_variance', 'g_n_peaks', 'r_n_peaks', 
            'g_rise_local_curvature', 'g_decline_local_curvature', 
            'r_rise_local_curvature', 'r_decline_local_curvature'
        ]
    
    def initialize_client(self) -> bool:
        """Initialize the ReLAISS client."""
        try:
            # Add path to ReLAISS
            sys.path.append('./laiss_final/re-laiss/src')
            
            import relaiss
            from relaiss.anomaly import anomaly_detection
            
            self.client = relaiss.ReLAISS()
            self.anomaly_detection = anomaly_detection
            
            logger.info("Loading ReLAISS reference with 25 light curve features...")
            self.client.load_reference(
                lc_features=self.lc_features,
                host_features=[],
                path_to_sfd_folder='./data/sfddata-master',
                force_recreation_of_index=True,
                building_for_AD=True, 
                num_trees=1000, 
                use_pca=False, 
                weight_lc=1.0 
            )
            
            self.initialized = True
            logger.info("ReLAISS anomaly detection client initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import ReLAISS: {e}")
            logger.error("Make sure ReLAISS is installed and accessible at './laiss_final/re-laiss/src'")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ReLAISS client: {e}")
            return False
    
    def run_anomaly_detection_on_object(self, ztfid: str) -> Optional[Dict]:
        """Run anomaly detection on a single object."""
        if not self.initialized:
            logger.warning("Anomaly detection client not initialized")
            return None
        
        try:
            logger.info(f"Running anomaly detection on {ztfid}")
            
            # Run anomaly detection
            result = self.anomaly_detection(
                client=self.client,
                transient_ztf_id=ztfid,
                lc_features=self.lc_features,
                host_features=[],
                path_to_timeseries_folder="./laiss_final/timeseries", 
                path_to_sfd_folder='./data/sfddata-master',         
                path_to_dataset_bank=self.client.bank_csv,
                path_to_models_directory="./laiss_final/models",       
                path_to_figure_directory="./laiss_final/figures/anomaly_detection", 
                save_figures=False,  # Don't save figures for now
                anom_thresh=self.anomaly_threshold,  
                force_retrain=True,
                preprocessed_df=None,
                return_scores=True,  # We want the scores back
                verbose=False
            )
            
            if result is not None:
                mjd_anom, anom_scores, norm_scores = result
                
                # Calculate overall anomaly score (max anomaly score)
                max_anomaly_score = float(max(anom_scores)) if anom_scores else 0.0
                
                return {
                    'ztfid': ztfid,
                    'anomaly_score': max_anomaly_score,
                    'mjd_anom': mjd_anom.tolist() if hasattr(mjd_anom, 'tolist') else list(mjd_anom),
                    'anom_scores': anom_scores.tolist() if hasattr(anom_scores, 'tolist') else list(anom_scores),
                    'norm_scores': norm_scores.tolist() if hasattr(norm_scores, 'tolist') else list(norm_scores),
                    'is_anomalous': max_anomaly_score > self.anomaly_threshold
                }
            
        except Exception as e:
            logger.error(f"Error running anomaly detection on {ztfid}: {e}")
            return None
    
    def run_anomaly_detection_batch(self, ztfids: List[str]) -> List[Dict]:
        """Run anomaly detection on a batch of objects."""
        if not self.initialized:
            if not self.initialize_client():
                logger.error("Could not initialize anomaly detection client")
                return []
        
        results = []
        for ztfid in ztfids:
            result = self.run_anomaly_detection_on_object(ztfid)
            if result:
                results.append(result)
        
        return results
    
    def process_new_objects_for_anomalies(self, db: Session, feature_extraction_run: models.FeatureExtractionRun) -> Optional[models.AnomalyNotification]:
        """
        Process all newly extracted objects for anomalies.
        
        Args:
            db: Database session
            feature_extraction_run: The feature extraction run that just completed
            
        Returns:
            AnomalyNotification if anomalies were found, None otherwise
        """
        if feature_extraction_run.status != "completed" or feature_extraction_run.objects_processed == 0:
            logger.info("No new objects to process for anomaly detection")
            return None
        
        # Get objects that were processed in this run
        # For simplicity, we'll check for objects that were updated recently
        recent_cutoff = feature_extraction_run.mjd_run - 1.0  # Within last day
        
        recent_objects = db.query(models.FeatureBank).filter(
            models.FeatureBank.mjd_extracted >= recent_cutoff
        ).all()
        
        if not recent_objects:
            logger.info("No recent objects found for anomaly detection")
            return None
        
        logger.info(f"Running anomaly detection on {len(recent_objects)} recently processed objects")
        
        ztfids_to_process = [obj.ztfid for obj in recent_objects]
        anomaly_results = self.run_anomaly_detection_batch(ztfids_to_process)
        
        # Store results in database
        anomalous_objects = []
        for result in anomaly_results:
            # Check if we already have a result for this object
            existing_result = db.query(models.AnomalyDetectionResult).filter(
                models.AnomalyDetectionResult.ztfid == result['ztfid']
            ).first()
            
            if existing_result:
                # Update existing result
                existing_result.anomaly_score = result['anomaly_score']
                existing_result.mjd_anom = result['mjd_anom']
                existing_result.anom_scores = result['anom_scores']
                existing_result.norm_scores = result['norm_scores']
                existing_result.detection_threshold = self.anomaly_threshold
                existing_result.is_anomalous = result['is_anomalous']
                existing_result.feature_extraction_run_id = feature_extraction_run.id
                existing_result.updated_at = datetime.utcnow()
            else:
                # Create new result
                anomaly_result = models.AnomalyDetectionResult(
                    ztfid=result['ztfid'],
                    anomaly_score=result['anomaly_score'],
                    mjd_anom=result['mjd_anom'],
                    anom_scores=result['anom_scores'],
                    norm_scores=result['norm_scores'],
                    detection_threshold=self.anomaly_threshold,
                    is_anomalous=result['is_anomalous'],
                    feature_extraction_run_id=feature_extraction_run.id
                )
                db.add(anomaly_result)
            
            # Track anomalous objects
            if result['is_anomalous']:
                anomalous_objects.append(result['ztfid'])
                logger.info(f"Anomalous object detected: {result['ztfid']} (score: {result['anomaly_score']:.1f}%)")
                
                # Create pending vote for anomalous science case
                create_pending_vote_for_science_case(
                    db, result['ztfid'], "anomalous", 
                    {"anomaly_detected": True, "anomaly_score": result['anomaly_score']}
                )
        
        db.commit()
        
        # Create notification if we found anomalies
        if anomalous_objects:
            notification = models.AnomalyNotification(
                detection_run_id=feature_extraction_run.id,
                objects_detected=len(anomalous_objects),
                ztfids_detected=anomalous_objects
            )
            db.add(notification)
            db.commit()
            
            logger.info(f"Created anomaly notification for {len(anomalous_objects)} objects: {anomalous_objects}")
            return notification
        
        return None
    
    def get_pending_anomalous_objects(self, db: Session) -> List[str]:
        """Get list of objects that are pending in the anomalous science case (legacy method)."""
        return get_pending_objects_for_science_case(db, "anomalous")
    
    def get_unacknowledged_notifications(self, db: Session) -> List[models.AnomalyNotification]:
        """Get list of unacknowledged anomaly notifications."""
        return db.query(models.AnomalyNotification).filter(
            models.AnomalyNotification.acknowledged == False
        ).order_by(models.AnomalyNotification.created_at.desc()).all()
    
    def acknowledge_notification(self, db: Session, notification_id: int):
        """Mark a notification as acknowledged."""
        notification = db.query(models.AnomalyNotification).filter(
            models.AnomalyNotification.id == notification_id
        ).first()
        
        if notification:
            notification.acknowledged = True
            notification.acknowledged_at = datetime.utcnow()
            db.commit()


# Global instance
anomaly_service = AnomalyDetectionService() 