"""
Slack Mode Service for reading recommendations from CSV files.

This service provides recommendations from pre-generated CSV files
instead of the real-time recommender system.
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from astropy.time import Time

logger = logging.getLogger(__name__)


class SlackModeService:
    """Service for handling Slack mode recommendations from CSV files."""
    
    def __init__(self, csv_directory: str = "data/slack_recommendations"):
        """
        Initialize the Slack mode service.
        
        Args:
            csv_directory: Directory containing the CSV files for each science case
        """
        self.csv_directory = csv_directory
        self._cache = {}  # Cache loaded CSV data
        self._cache_timestamp = {}  # Track when each file was last loaded
        self.cache_duration_hours = 1  # Reload CSV files every hour
        
    def _get_csv_path(self, science_case: str) -> str:
        """Get the path to the CSV file for a given science case."""
        # Map science cases to CSV filenames
        filename_map = {
            "snia-like": "snia-like.csv",
            "ccsn-like": "ccsn-like.csv", 
            "long-lived": "long-lived.csv",
            "anomalous": "anomalous.csv",
            "precursor": "precursor.csv",
            "all": "all.csv"
        }
        
        filename = filename_map.get(science_case, f"{science_case}_recommendations.csv")
        return os.path.join(self.csv_directory, filename)
    
    def _should_reload_csv(self, science_case: str) -> bool:
        """Check if we should reload the CSV file."""
        if science_case not in self._cache_timestamp:
            return True
            
        last_loaded = self._cache_timestamp[science_case]
        time_since_load = datetime.utcnow() - last_loaded
        
        return time_since_load > timedelta(hours=self.cache_duration_hours)
    
    def _load_csv(self, science_case: str) -> pd.DataFrame:
        """Load CSV file for a science case, with caching."""
        if not self._should_reload_csv(science_case) and science_case in self._cache:
            logger.info(f"Using cached CSV data for {science_case}")
            return self._cache[science_case]
        
        csv_path = self._get_csv_path(science_case)
        
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
        
        try:
            logger.info(f"Loading CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Ensure required columns exist
            required_columns = ['ztfid', 'recommendation_mjd']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' not found in {csv_path}")
                    return pd.DataFrame()
            
            # Cache the data
            self._cache[science_case] = df
            self._cache_timestamp[science_case] = datetime.utcnow()
            
            logger.info(f"Loaded {len(df)} recommendations for {science_case}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            return pd.DataFrame()
    
    def get_recommendations(
        self,
        science_case: str,
        lookback_days: float,
        excluded_ztfids: List[str],
        count: int = 10,
        start_ztfid: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recommendations from CSV files.
        
        Args:
            science_case: The science case to get recommendations for
            lookback_days: How many days back to look for recommendations
            excluded_ztfids: List of ZTFIDs to exclude (already voted on)
            count: Number of recommendations to return
            start_ztfid: Optional starting ZTFID
            
        Returns:
            List of recommendation dictionaries
        """
        df = self._load_csv(science_case)
        
        if df.empty:
            logger.warning(f"No CSV data available for {science_case}")
            return []
        
        # Calculate the cutoff MJD
        current_mjd = Time.now().mjd
        cutoff_mjd = current_mjd - lookback_days
        
        logger.info(f"Filtering recommendations: current_mjd={current_mjd:.2f}, cutoff_mjd={cutoff_mjd:.2f}, lookback_days={lookback_days}")
        
        # Filter by lookback period
        mask = df['recommendation_mjd'] >= cutoff_mjd
        filtered_df = df[mask].copy()
        
        logger.info(f"Found {len(filtered_df)} recommendations within {lookback_days} days")
        
        # Exclude already voted objects
        if excluded_ztfids:
            mask = ~filtered_df['ztfid'].isin(excluded_ztfids)
            filtered_df = filtered_df[mask]
            logger.info(f"After excluding voted objects: {len(filtered_df)} recommendations")
        
        # Handle start_ztfid if provided
        if start_ztfid and start_ztfid in filtered_df['ztfid'].values:
            start_idx = filtered_df[filtered_df['ztfid'] == start_ztfid].index[0]
            filtered_df = filtered_df.loc[start_idx:]
            logger.info(f"Starting from {start_ztfid}, {len(filtered_df)} recommendations remaining")
        
        # Sort by recommendation date (newest first) and take requested count
        filtered_df = filtered_df.sort_values('recommendation_mjd', ascending=False)
        result_df = filtered_df.head(count)
        
        # Convert to list of dictionaries
        recommendations = []
        for _, row in result_df.iterrows():
            # Handle ZTFID - ensure it has the ZTF prefix
            # Accept both ZTF and ANTARES IDs
            ztfid = str(row['ztfid'])
            if not ztfid.startswith('ZTF') and not ztfid.startswith('ANT'):
                ztfid = f'ZTF{ztfid}'
            
            rec = {
                'ZTFID': ztfid,
                'science_case': science_case,
                'recommendation_mjd': float(row['recommendation_mjd']),
                'is_slack_mode': True  # Flag to indicate this is from Slack mode
            }
            
            # Add optional fields if they exist in the CSV
            optional_fields = ['ra', 'dec', 'latest_magnitude', 'score', 'comment']
            for field in optional_fields:
                if field in row and pd.notna(row[field]):
                    rec[field] = float(row[field]) if field in ['ra', 'dec', 'latest_magnitude', 'score'] else str(row[field])
            
            recommendations.append(rec)
        
        logger.info(f"Returning {len(recommendations)} recommendations for {science_case}")
        return recommendations
    
    def check_csv_status(self) -> Dict[str, Dict]:
        """Check the status of CSV files for all science cases."""
        status = {}
        
        science_cases = ["snia-like", "ccsn-like", "long-lived", "anomalous", "precursor"]
        
        for science_case in science_cases:
            csv_path = self._get_csv_path(science_case)
            
            if os.path.exists(csv_path):
                try:
                    stat = os.stat(csv_path)
                    df = self._load_csv(science_case)
                    
                    status[science_case] = {
                        'exists': True,
                        'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'file_size_kb': stat.st_size / 1024,
                        'total_recommendations': len(df) if not df.empty else 0,
                        'latest_recommendation_mjd': float(df['recommendation_mjd'].max()) if not df.empty and 'recommendation_mjd' in df.columns else None
                    }
                except Exception as e:
                    logger.error(f"Error checking status for {science_case}: {e}")
                    status[science_case] = {
                        'exists': True,
                        'error': str(e)
                    }
            else:
                status[science_case] = {
                    'exists': False,
                    'path': csv_path
                }
        
        return status 