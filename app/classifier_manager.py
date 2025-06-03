"""Classifier configuration and management system."""

import yaml
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session

from . import models
from .pending_votes import create_pending_vote_for_science_case
from .anomaly_service import anomaly_service

logger = logging.getLogger(__name__)

class ClassifierManager:
    """Manages classifier configurations and execution."""
    
    def __init__(self, config_path: str = "app/classifier_config.yaml"):
        """Initialize the classifier manager."""
        self.config_path = Path(config_path)
        self.config = None
        self.load_config()
    
    def load_config(self) -> bool:
        """Load classifier configuration from YAML file."""
        try:
            if not self.config_path.exists():
                logger.error(f"Classifier config file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Loaded classifier configuration from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading classifier config: {e}")
            return False
    
    def get_enabled_classifiers(self, science_case: str) -> List[Dict[str, Any]]:
        """Get list of enabled classifiers for a science case."""
        if not self.config:
            return []
        
        classifiers = self.config.get('classifiers', {}).get(science_case, [])
        return [c for c in classifiers if c.get('enabled', False)]
    
    def get_classifier_info(self, science_case: str, classifier_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific classifier."""
        if not self.config:
            return None
        
        classifiers = self.config.get('classifiers', {}).get(science_case, [])
        for classifier in classifiers:
            if classifier.get('name') == classifier_name:
                return classifier
        
        return None
    
    def run_classifiers_for_science_case(self, db: Session, science_case: str, 
                                       feature_extraction_run: models.FeatureExtractionRun) -> List[str]:
        """
        Run all enabled classifiers for a science case.
        
        Returns:
            List of ZTFIDs that were classified as pending for this science case
        """
        if science_case == "anomalous":
            # Special handling for anomalous - use existing anomaly service
            notification = anomaly_service.process_new_objects_for_anomalies(db, feature_extraction_run)
            if notification and notification.ztfids_detected:
                return notification.ztfids_detected
            return []
        
        enabled_classifiers = self.get_enabled_classifiers(science_case)
        if not enabled_classifiers:
            logger.info(f"No enabled classifiers for {science_case}")
            return []
        
        classified_objects = []
        
        for classifier_config in enabled_classifiers:
            try:
                # TODO: Implement actual classifier execution
                # For now, return empty list as placeholders
                logger.info(f"TODO: Execute {classifier_config['name']} for {science_case}")
                
                # Placeholder - would run the actual classifier here
                # module_name = classifier_config['module']
                # function_name = classifier_config['function']
                # module = importlib.import_module(module_name)
                # classifier_function = getattr(module, function_name)
                # results = classifier_function(db, feature_extraction_run)
                
                # For now, just log the configuration
                logger.info(f"Classifier config: {classifier_config}")
                
            except Exception as e:
                logger.error(f"Error running classifier {classifier_config['name']}: {e}")
        
        return classified_objects
    
    def create_placeholder_pending_votes(self, db: Session) -> None:
        """
        Create placeholder pending votes for all science cases except anomalous.
        This ensures the system works out of the box with empty pending lists.
        """
        science_cases = ["snia-like", "ccsn-like", "long-lived", "precursor"]
        
        for science_case in science_cases:
            try:
                # Check if there are already pending votes for this science case
                existing_pending = db.query(models.PendingVote).filter(
                    models.PendingVote.science_case == science_case
                ).first()
                
                if not existing_pending:
                    logger.info(f"No pending votes found for {science_case} - system ready for future classifiers")
                    # Note: We don't create dummy pending votes, just ensure the system can handle empty lists
                
            except Exception as e:
                logger.error(f"Error checking pending votes for {science_case}: {e}")
    
    def get_classifier_badge_info(self, db: Session, ztfid: str) -> List[Dict[str, Any]]:
        """
        Get classifier badge information for a ZTFID.
        
        Returns:
            List of badge information with classifier name, confidence, description, and URL
        """
        badges = []
        
        # Check for anomaly detection results
        anomaly_result = db.query(models.AnomalyDetectionResult).filter(
            models.AnomalyDetectionResult.ztfid == ztfid
        ).first()
        
        if anomaly_result and anomaly_result.is_anomalous:
            classifier_info = self.get_classifier_info("anomalous", "reLAISS")
            if classifier_info:
                # Count epochs with data
                num_epochs = len(anomaly_result.anom_scores) if anomaly_result.anom_scores else 0
                
                badges.append({
                    'classifier_name': 'reLAISS',
                    'classifier_url': classifier_info['url'],
                    'confidence': anomaly_result.anomaly_score,
                    'num_epochs': num_epochs,
                    'description': classifier_info['description'],
                    'badge_text': f"This transient was flagged by reLAISS with {anomaly_result.anomaly_score:.1f}% anomaly score across {num_epochs} epochs.",
                    'badge_type': 'anomaly'
                })
        
        # TODO: Add checks for other classifier results
        # This is where we would check for SUPERPHOT+, RAPID, etc. results
        # Example structure:
        # snia_result = db.query(models.SNIaClassificationResult).filter(...)
        # if snia_result:
        #     badges.append({
        #         'classifier_name': 'SUPERPHOT+',
        #         'classifier_url': 'https://github.com/LSSTDESC/superphot-plus',
        #         'confidence': snia_result.confidence,
        #         'classification': snia_result.predicted_class,
        #         'description': 'SN Ia classification using photometric features',
        #         'badge_text': f"This transient was classified by SUPERPHOT+ as {snia_result.predicted_class} with {snia_result.confidence:.1f}% probability.",
        #         'badge_type': 'classification'
        #     })
        
        return badges

# Global instance
classifier_manager = ClassifierManager() 