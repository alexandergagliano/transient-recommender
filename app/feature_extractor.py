"""Feature extraction module for transient objects."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from astropy.time import Time
from astropy import units as u
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from multiprocessing import Pool, cpu_count
from functools import partial
import requests
import warnings
from tqdm import tqdm
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import find_peaks
import json
import traceback

from . import models
from .filter_manager import filter_manager

# Optional imports with error handling
try:
    import antares_client
    from antares_client.search import search
except ImportError:
    antares_client = None
    search = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def execute_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """Execute a function with a timeout using threading instead of signals."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            # The thread will continue running but we abandon waiting for it
            raise TimeoutError(f"Function execution timed out after {timeout_seconds} seconds")

class ALeRCEFetcher:
    """
    Fetcher for recent transients from ALeRCE API.
    Much more reliable than ANTARES for bulk operations.
    """
    BASE_URL = "https://api.alerce.online/ztf/v1"

    def __init__(self, days=20, max_pages=10, page_size=1000, max_threads=8):
        self.days = days
        self.max_pages = max_pages
        self.page_size = page_size
        self.max_threads = max_threads
        self.mjd_start, self.mjd_end = self._get_mjd_range()
        self.logger = logging.getLogger(__name__)

    def _get_mjd_range(self):
        """Convert date range to MJD for ALeRCE API"""
        now = datetime.utcnow()
        start = now - timedelta(days=self.days)
        return self._to_mjd(start), self._to_mjd(now)

    def _to_mjd(self, dt):
        """Convert datetime to MJD"""
        return dt.timestamp() / 86400.0 + 40587

    def fetch_recent_oids(self):
        """Quickly fetch all unique object IDs with detections in the date range."""
        self.logger.info(f"Fetching recent objects from ALeRCE (last {self.days} days)")
        self.logger.info(f"MJD range: {self.mjd_start:.2f} to {self.mjd_end:.2f}")
        
        oids = set()
        for page in range(1, self.max_pages + 1):
            try:
                self.logger.debug(f"Fetching page {page}/{self.max_pages}")
                resp = requests.get(
                    f"{self.BASE_URL}/objects",
                    params={
                        "first_mjd__gte": self.mjd_start,
                        "first_mjd__lte": self.mjd_end,
                        "page": page,
                        "page_size": self.page_size,
                    },
                    timeout=30
                )
                
                if resp.status_code != 200:
                    self.logger.warning(f"Page {page} returned status {resp.status_code}")
                    break
                    
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    self.logger.info(f"No more results at page {page}")
                    break
                    
                page_oids = set()
                for obj in results:
                    if "oid" in obj:
                        page_oids.add(obj["oid"])
                
                oids.update(page_oids)
                self.logger.info(f"Page {page}: +{len(page_oids)} objects (total: {len(oids)})")
                
                # Stop if we got fewer results than page size (last page)
                if len(results) < self.page_size:
                    self.logger.info(f"Reached end of results at page {page}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break
        
        self.logger.info(f"Found {len(oids)} unique objects with recent detections")
        return list(oids)

    def _get_object_info(self, oid):
        """Get object classification and basic info"""
        try:
            resp = requests.get(f"{self.BASE_URL}/objects/{oid}", timeout=15)
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            # Extract key information
            info = {
                'oid': oid,
                'ra': data.get('meanra'),
                'dec': data.get('meandec'),
                'classification': data.get('classifier', {}),
                'n_detections': data.get('ndet', 0),
                'first_mjd': data.get('firstmjd'),
                'last_mjd': data.get('lastmjd')
            }
            
            return info
            
        except Exception as e:
            self.logger.debug(f"Error getting info for {oid}: {e}")
            return None

    def _is_interesting_transient(self, obj_info):
        """
        Determine if object is an interesting transient for our purposes.
        Focus on supernovae and other transients, skip variables.
        """
        if not obj_info:
            return False
            
        classification = obj_info.get('classification', {})
        classifier = classification.get('classifier', '')
        
        # Accept SN classifications
        if classifier in ['SN', 'SNIa', 'SNIbc', 'SNII']:
            return True
            
        # Accept high-confidence transient classifications
        prob = classification.get('probability', 0)
        if classifier in ['Transient', 'Unknown'] and prob > 0.7:
            return True
            
        # Require minimum number of detections
        if obj_info.get('n_detections', 0) < 5:
            return False
            
        # Skip obvious variables
        if classifier in ['Variable', 'Periodic', 'RRLyr', 'EclBin']:
            return False
            
        return True

    def get_recent_transients(self):
        """
        Get recent interesting transients with their basic info.
        Returns a list of objects suitable for feature extraction.
        """
        # Step 1: Get all recent object IDs
        all_oids = self.fetch_recent_oids()
        
        if not all_oids:
            self.logger.warning("No recent objects found")
            return []
        
        # Limit to reasonable number for processing
        if len(all_oids) > 500:
            self.logger.info(f"Limiting to 500 most recent objects (found {len(all_oids)})")
            all_oids = all_oids[:500]
        
        # Step 2: Get object info in parallel
        self.logger.info(f"Getting classification info for {len(all_oids)} objects...")
        
        interesting_objects = []
        
        def process_oid(oid):
            obj_info = self._get_object_info(oid)
            if obj_info and self._is_interesting_transient(obj_info):
                return obj_info
            return None
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_oid = {executor.submit(process_oid, oid): oid for oid in all_oids}
            
            completed = 0
            for future in as_completed(future_to_oid):
                completed += 1
                if completed % 50 == 0:
                    self.logger.info(f"Processed {completed}/{len(all_oids)} objects...")
                
                result = future.result()
                if result:
                    interesting_objects.append(result)
        
        self.logger.info(f"Found {len(interesting_objects)} interesting transients")
        
        # Sort by last detection time (most recent first)
        interesting_objects.sort(key=lambda x: x.get('last_mjd', 0), reverse=True)
        
        return interesting_objects

def get_daily_objects(lookback_days: float = 20.0, lookback_t_first: float = 500, test_mode: bool = False) -> Optional[pd.DataFrame]:
    """
    Get objects with detections in the last N days using ALeRCE API.
    
    This replaces the unreliable ANTARES approach with ALeRCE bulk queries.
    
    Args:
        lookback_days: Number of days to look back
        lookback_t_first: Not used in current implementation
        test_mode: If True, skip query entirely and return None gracefully
    """
    logger.info(f"Getting objects with detections in the last {lookback_days} days using ALeRCE (test_mode={test_mode})")
    
    if test_mode:
        logger.info("Test mode enabled - skipping ALeRCE query")
        return None
    
    try:
        # Initialize ALeRCE fetcher
        fetcher = ALeRCEFetcher(days=lookback_days, max_pages=5, max_threads=8)
        
        # Get recent transients
        transients = fetcher.get_recent_transients()
        
        if not transients:
            logger.warning("No interesting transients found in ALeRCE")
            return None
        
        # Convert to DataFrame format expected by the rest of the system
        objects_data = []
        
        for obj in transients:
            oid = obj['oid']
            
            # Convert ALeRCE OID to ZTFID format if needed
            if not oid.startswith('ZTF'):
                # ALeRCE uses ZTF object IDs, but might not have ZTF prefix
                ztfid = f"ZTF{oid}" if oid.startswith('1') else oid
            else:
                ztfid = oid
            
            ra = obj.get('ra')
            dec = obj.get('dec')
            last_mjd = obj.get('last_mjd', 0)
            
            if ra is None or dec is None:
                logger.warning(f"Missing coordinates for {ztfid}")
                continue
            
            objects_data.append({
                'ZTFID': ztfid,
                'ra': float(ra),
                'dec': float(dec),
                'newest_alert': float(last_mjd),
                'alerce_info': obj  # Store full ALeRCE info for later use
            })
        
        if not objects_data:
            logger.warning("No valid objects with coordinates found")
            return None
        
        df = pd.DataFrame(objects_data)
        logger.info(f"Successfully retrieved {len(df)} transients from ALeRCE")
        
        # Add status information
        logger.info("ALeRCE Query Status:")
        logger.info(f"• Successfully retrieved {len(df)} recent transients")
        logger.info("• Used reliable ALeRCE API instead of problematic ANTARES")
        logger.info("• Filtered for interesting transients (SNe, high-confidence unknowns)")
        logger.info("• Objects ready for feature extraction")
        
        return df
        
    except Exception as e:
        logger.error(f"Error querying ALeRCE: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.info("ALeRCE Status Summary:")
        logger.info("• ALeRCE API query failed")
        logger.info("• New object discovery is temporarily unavailable")
        logger.info("• Existing objects in database will continue to be processed")
        logger.info("• Manual object addition via web interface still works")
        logger.info("• System will automatically retry on next scheduled run")
        return None

def extract_light_curve_features(timeseries: pd.DataFrame) -> Dict:
    """
    Extract light curve features from a timeseries DataFrame.
    
    This implements simplified light curve feature extraction focused on photometric properties.
    """
    features = {}
    
    try:
        # Separate by filter
        g_data = timeseries[timeseries['ant_passband'] == 'g'].copy()
        r_data = timeseries[timeseries['ant_passband'] == 'r'].copy()
        
        # Calculate latest magnitude from all data
        if len(timeseries) > 0:
            latest_idx = timeseries['ant_mjd'].idxmax()
            features['latest_magnitude'] = timeseries.loc[latest_idx, 'ant_mag']
            features['newest_alert'] = timeseries.loc[latest_idx, 'ant_mjd']
        else:
            features['latest_magnitude'] = np.nan
            features['newest_alert'] = np.nan
        
        # Basic statistics for each filter
        for filt, data in [('g', g_data), ('r', r_data)]:
            if len(data) == 0:
                # No data for this filter - set all features to NaN
                features[f'{filt}_peak_mag'] = np.nan
                features[f'{filt}_peak_time'] = np.nan
                features[f'{filt}_rise_time'] = np.nan
                features[f'{filt}_decline_time'] = np.nan
                features[f'{filt}_duration_above_half_flux'] = np.nan
                features[f'{filt}_mean_rolling_variance'] = np.nan
                features[f'{filt}_rise_local_curvature'] = np.nan
                features[f'{filt}_decline_local_curvature'] = np.nan
                continue
            
            # Sort by time
            data = data.sort_values('ant_mjd').copy()
            
            # Peak magnitude (brightest = lowest magnitude)
            peak_idx = data['ant_mag'].idxmin()
            features[f'{filt}_peak_mag'] = data.loc[peak_idx, 'ant_mag']
            features[f'{filt}_peak_time'] = data.loc[peak_idx, 'ant_mjd']
            
            # Rise and decline times
            peak_mjd = features[f'{filt}_peak_time']
            pre_peak = data[data['ant_mjd'] <= peak_mjd]
            post_peak = data[data['ant_mjd'] >= peak_mjd]
            
            if len(pre_peak) > 1:
                features[f'{filt}_rise_time'] = peak_mjd - pre_peak['ant_mjd'].min()
            else:
                features[f'{filt}_rise_time'] = np.nan
            
            if len(post_peak) > 1:
                features[f'{filt}_decline_time'] = post_peak['ant_mjd'].max() - peak_mjd
            else:
                features[f'{filt}_decline_time'] = np.nan
            
            # Duration above half maximum flux
            peak_mag = features[f'{filt}_peak_mag']
            half_max_mag = peak_mag + 0.75  # 0.75 mag fainter = half flux
            above_half = data[data['ant_mag'] <= half_max_mag]
            
            if len(above_half) > 1:
                features[f'{filt}_duration_above_half_flux'] = above_half['ant_mjd'].max() - above_half['ant_mjd'].min()
            else:
                features[f'{filt}_duration_above_half_flux'] = np.nan
            
            # Rolling variance (simplified)
            if len(data) >= 5:
                rolling_var = data['ant_mag'].rolling(window=5, center=True).var()
                features[f'{filt}_mean_rolling_variance'] = rolling_var.mean()
            else:
                features[f'{filt}_mean_rolling_variance'] = data['ant_mag'].var() if len(data) > 1 else np.nan
            
            # Local curvature (simplified - second derivative)
            if len(data) >= 3:
                try:
                    mjd_range = data['ant_mjd'].max() - data['ant_mjd'].min()
                    if mjd_range > 0:
                        mjd_grid = np.linspace(data['ant_mjd'].min(), data['ant_mjd'].max(), min(len(data), 20))
                        mag_interp = interp1d(data['ant_mjd'], data['ant_mag'], kind='linear', fill_value='extrapolate')
                        mag_grid = mag_interp(mjd_grid)
                        
                        # Second derivative
                        if len(mjd_grid) >= 3:
                            d2mag = np.gradient(np.gradient(mag_grid, mjd_grid), mjd_grid)
                            
                            # Rise curvature (before peak)
                            peak_idx_grid = np.argmin(mag_grid)
                            if peak_idx_grid > 0:
                                features[f'{filt}_rise_local_curvature'] = np.mean(d2mag[:peak_idx_grid])
                            else:
                                features[f'{filt}_rise_local_curvature'] = np.nan
                            
                            # Decline curvature (after peak)
                            if peak_idx_grid < len(d2mag) - 1:
                                features[f'{filt}_decline_local_curvature'] = np.mean(d2mag[peak_idx_grid+1:])
                            else:
                                features[f'{filt}_decline_local_curvature'] = np.nan
                        else:
                            features[f'{filt}_rise_local_curvature'] = np.nan
                            features[f'{filt}_decline_local_curvature'] = np.nan
                    else:
                        features[f'{filt}_rise_local_curvature'] = np.nan
                        features[f'{filt}_decline_local_curvature'] = np.nan
                except Exception as e:
                    logger.warning(f"Error calculating curvature for {filt}: {e}")
                    features[f'{filt}_rise_local_curvature'] = np.nan
                    features[f'{filt}_decline_local_curvature'] = np.nan
            else:
                features[f'{filt}_rise_local_curvature'] = np.nan
                features[f'{filt}_decline_local_curvature'] = np.nan
        
        # Color features (if both g and r data exist)
        if len(g_data) > 0 and len(r_data) > 0:
            try:
                # Interpolate to common time grid for color calculation
                min_mjd = max(g_data['ant_mjd'].min(), r_data['ant_mjd'].min())
                max_mjd = min(g_data['ant_mjd'].max(), r_data['ant_mjd'].max())
                
                if max_mjd > min_mjd:
                    common_mjd = np.linspace(min_mjd, max_mjd, 10)
                    
                    g_interp = interp1d(g_data['ant_mjd'], g_data['ant_mag'], kind='linear', fill_value='extrapolate')
                    r_interp = interp1d(r_data['ant_mjd'], r_data['ant_mag'], kind='linear', fill_value='extrapolate')
                    
                    g_common = g_interp(common_mjd)
                    r_common = r_interp(common_mjd)
                    
                    g_minus_r = g_common - r_common
                    
                    features['mean_g-r'] = np.mean(g_minus_r)
                    
                    # Color at g peak
                    g_peak_time = features.get('g_peak_time')
                    if not np.isnan(g_peak_time) and min_mjd <= g_peak_time <= max_mjd:
                        g_at_peak = g_interp(g_peak_time)
                        r_at_peak = r_interp(g_peak_time)
                        features['g-r_at_g_peak'] = g_at_peak - r_at_peak
                    else:
                        features['g-r_at_g_peak'] = np.nan
                    
                    # Color evolution rate
                    if len(common_mjd) > 1:
                        color_rate = np.gradient(g_minus_r, common_mjd)
                        features['mean_color_rate'] = np.mean(np.abs(color_rate))
                    else:
                        features['mean_color_rate'] = np.nan
                else:
                    features['mean_g-r'] = np.nan
                    features['g-r_at_g_peak'] = np.nan
                    features['mean_color_rate'] = np.nan
            except Exception as e:
                logger.warning(f"Error calculating color features: {e}")
                features['mean_g-r'] = np.nan
                features['g-r_at_g_peak'] = np.nan
                features['mean_color_rate'] = np.nan
        else:
            features['mean_g-r'] = np.nan
            features['g-r_at_g_peak'] = np.nan
            features['mean_color_rate'] = np.nan
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting light curve features: {e}")
        return {}

def process_object(obj_row: pd.Series) -> Optional[pd.DataFrame]:
    """
    Process a single object to extract light curve features.
    
    This fetches the light curve from ALeRCE and extracts features.
    """
    ztfid = obj_row.get('ZTFID', 'unknown')
    
    try:
        # Get light curve from ALeRCE API
        timeseries_data = []
        
        # Extract OID from ZTFID for ALeRCE API
        oid = ztfid
        if ztfid.startswith('ZTF'):
            # Remove ZTF prefix if present
            oid = ztfid[3:] if len(ztfid) > 3 else ztfid
        
        try:
            # Fetch light curve from ALeRCE
            resp = requests.get(f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve", timeout=30)
            
            if resp.status_code != 200:
                logger.warning(f"ALeRCE API returned status {resp.status_code} for {ztfid}")
                return None
            
            data = resp.json()
            lc_data = data.get("detections", [])
            
            if not lc_data:
                logger.warning(f"No light curve data found for {ztfid}")
                return None
            
            # Convert ALeRCE format to our internal format
            for point in lc_data:
                # ALeRCE uses different field names
                mjd = point.get('mjd')
                mag = point.get('magpsf_corr') or point.get('magpsf')  # Use corrected mag if available
                magerr = point.get('sigmapsf_corr') or point.get('sigmapsf', 0.1)  # Use corrected error if available
                fid = point.get('fid')  # Filter ID (1=g, 2=r, 3=i)
                
                # Convert filter ID to passband name
                passband_map = {1: 'g', 2: 'r', 3: 'i'}
                passband = passband_map.get(fid, 'unknown')
                
                # Only include valid detections (skip non-detections)
                if mjd and mag and not np.isnan(mag) and mag > 0 and point.get('isdiffpos', 1) != 0:
                    timeseries_data.append({
                        'ant_mjd': mjd,
                        'ant_mag': mag,
                        'ant_magerr': magerr,
                        'ant_passband': passband
                    })
            
        except (requests.exceptions.RequestException, Exception) as e:
            logger.warning(f"Failed to fetch light curve for {ztfid}: {e}")
            return None
        
        if not timeseries_data:
            logger.warning(f"No valid timeseries data found for {ztfid}")
            return None
        
        timeseries = pd.DataFrame(timeseries_data)
        
        # Extract features
        features = extract_light_curve_features(timeseries)
        
        if not features:
            logger.warning(f"No features extracted for {ztfid}")
            return None
        
        # Add basic object info
        features['ZTFID'] = ztfid
        features['ra'] = obj_row.get('ra')
        features['dec'] = obj_row.get('dec')
        
        # Add latest magnitude info
        if len(timeseries_data) > 0:
            # Sort by MJD to get latest
            sorted_data = sorted(timeseries_data, key=lambda x: x['ant_mjd'])
            features['latest_magnitude'] = sorted_data[-1]['ant_mag']
        
        # Create feature errors (simplified - use 10% of feature value or default)
        feature_errors = {}
        for key, value in features.items():
            if key in ['ZTFID', 'ra', 'dec', 'latest_magnitude']:
                continue
            if np.isnan(value) or value == 0:
                feature_errors[f'{key}_err'] = np.nan
            else:
                feature_errors[f'{key}_err'] = abs(value) * 0.1  # 10% error
        
        # Combine features and errors
        all_features = {**features, **feature_errors}
        
        # Convert to DataFrame
        result_df = pd.DataFrame([all_features])
        
        logger.info(f"Successfully processed {ztfid} with {len(timeseries_data)} data points")
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing object {ztfid}: {e}")
        return None

def extract_features_for_recent_objects(
    db: Session, 
    lookback_days: float = 20.0,
    force_reprocess: bool = False,
    test_mode: bool = False
) -> models.FeatureExtractionRun:
    """
    Extract features for objects detected in the last N days.
    
    This implements the feature extraction pipeline from orig_recommender.py
    """
    start_time = time.time()
    today_mjd = Time.now().mjd
    
    # Create a new extraction run record
    extraction_run = models.FeatureExtractionRun(
        mjd_run=today_mjd,
        lookback_days=lookback_days,
        status="running"
    )
    db.add(extraction_run)
    db.commit()
    
    try:
        logger.info(f"Starting feature extraction for objects in last {lookback_days} days")
        
        # Get the last successful run to determine actual lookback period
        last_run = db.query(models.FeatureExtractionRun).filter(
            models.FeatureExtractionRun.status == "completed"
        ).order_by(models.FeatureExtractionRun.mjd_run.desc()).first()
        
        if last_run and not force_reprocess:
            # Calculate days since last run
            days_since_last = today_mjd - last_run.mjd_run
            actual_lookback = min(lookback_days, days_since_last + 1)  # Add 1 day buffer
            logger.info(f"Last run was {days_since_last:.1f} days ago, using lookback of {actual_lookback:.1f} days")
        else:
            actual_lookback = lookback_days
            logger.info(f"No previous run found or force reprocess, using full lookback of {actual_lookback:.1f} days")
        
        # Get objects with recent detections
        daily_objects = get_daily_objects(lookback_days=actual_lookback, test_mode=test_mode)
        
        if daily_objects is None or len(daily_objects) == 0:
            if test_mode:
                logger.info("Test mode active - no objects processed (this is expected)")
            else:
                logger.warning("No new objects found from ALeRCE query")
            extraction_run.objects_found = 0
            extraction_run.objects_processed = 0
            extraction_run.status = "completed"
            extraction_run.completed_at = datetime.utcnow()
            extraction_run.processing_time_seconds = time.time() - start_time
            db.commit()
            return extraction_run
        
        extraction_run.objects_found = len(daily_objects)
        db.commit()
        
        # Get existing feature bank to check what needs updating
        existing_features = db.query(models.FeatureBank).all()
        existing_ztfids = {f.ztfid: f.mjd_extracted for f in existing_features}
        
        # Determine which objects need feature extraction
        objects_to_process = []
        for _, obj in daily_objects.iterrows():
            ztfid = obj['ZTFID']
            
            # Check if we need to process this object
            if ztfid not in existing_ztfids:
                # New object, needs processing
                objects_to_process.append(obj)
                logger.debug(f"New object: {ztfid}")
            elif force_reprocess or (today_mjd - existing_ztfids[ztfid]) > 2:
                # Existing object but features are stale (>2 days old)
                objects_to_process.append(obj)
                logger.debug(f"Stale features for: {ztfid}")
        
        logger.info(f"Processing features for {len(objects_to_process)} objects")
        
        # Process objects using multiprocessing (like in original)
        processed_count = 0
        
        if len(objects_to_process) > 0:
            # Use multiprocessing for efficiency
            max_workers = min(8, cpu_count())
            logger.info(f"Using {max_workers} workers for feature extraction")
            
            with Pool(processes=max_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(process_object, [row for _, row in pd.DataFrame(objects_to_process).iterrows()]),
                        total=len(objects_to_process),
                        desc="Extracting features"
                    )
                )
            
            # Process results
            for i, features_df in enumerate(results):
                if features_df is not None:
                    try:
                        obj = objects_to_process[i]
                        ztfid = obj['ZTFID']
                        
                        # Check if object already exists in database
                        existing_obj = db.query(models.FeatureBank).filter(
                            models.FeatureBank.ztfid == ztfid
                        ).first()
                        
                        # Prepare feature data
                        feature_row = features_df.iloc[0]
                        
                        # Separate features and errors
                        features_dict = {}
                        errors_dict = {}
                        
                        for col in feature_row.index:
                            if col.endswith('_err'):
                                errors_dict[col] = feature_row[col]
                            elif col not in ['ZTFID', 'ra', 'dec']:
                                features_dict[col] = feature_row[col]
                        
                        if existing_obj:
                            # Update existing
                            existing_obj.features = features_dict
                            existing_obj.feature_errors = errors_dict
                            existing_obj.ra = feature_row['ra']
                            existing_obj.dec = feature_row['dec']
                            existing_obj.latest_magnitude = feature_row.get('latest_magnitude')
                            existing_obj.mjd_extracted = today_mjd
                            existing_obj.last_updated = datetime.utcnow()
                        else:
                            # Create new
                            new_obj = models.FeatureBank(
                                ztfid=ztfid,
                                ra=feature_row['ra'],
                                dec=feature_row['dec'],
                                latest_magnitude=feature_row.get('latest_magnitude'),
                                features=features_dict,
                                feature_errors=errors_dict,
                                mjd_extracted=today_mjd
                            )
                            db.add(new_obj)
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error saving features for object {i}: {e}")
                        continue
        
        # Update extraction run
        extraction_run.objects_processed = processed_count
        extraction_run.status = "completed"
        extraction_run.completed_at = datetime.utcnow()
        extraction_run.processing_time_seconds = time.time() - start_time
        
        db.commit()
        
        logger.info(f"Feature extraction completed: {processed_count}/{len(objects_to_process)} objects processed in {extraction_run.processing_time_seconds:.1f}s")
        
        # Run classifiers on newly processed objects using the classifier manager
        try:
            logger.info("Running automated classifiers on newly processed objects...")
            
            # Run classifiers for all science cases
            science_cases = ["anomalous", "snia-like", "ccsn-like", "long-lived", "precursor"]
            total_classified = 0
            
            for science_case in science_cases:
                try:
                    filtered_objects = filter_manager.run_filters_for_science_case(
                        db, science_case, extraction_run
                    )
                    if filtered_objects:
                        logger.info(f"Filtered {len(filtered_objects)} objects for {science_case}: {filtered_objects}")
                        total_classified += len(filtered_objects)
                    else:
                        logger.info(f"No objects filtered for {science_case}")
                except Exception as e:
                    logger.error(f"Error running filters for {science_case}: {e}")
            
            logger.info(f"Filter processing completed: {total_classified} total objects filtered across all science cases")
            
            # Initialize placeholder pending votes to ensure system works out of the box
            filter_manager.create_placeholder_pending_votes(db)
            
        except Exception as e:
            logger.error(f"Error running automated filters: {e}")
            # Don't fail the entire extraction run if filter processing fails
        
        return extraction_run
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        extraction_run.status = "failed"
        extraction_run.error_message = str(e)
        extraction_run.completed_at = datetime.utcnow()
        extraction_run.processing_time_seconds = time.time() - start_time
        db.commit()
        raise

def get_last_extraction_run(db: Session) -> Optional[models.FeatureExtractionRun]:
    """Get the most recent feature extraction run."""
    return db.query(models.FeatureExtractionRun).order_by(
        models.FeatureExtractionRun.mjd_run.desc()
    ).first()

def should_run_feature_extraction(db: Session, max_age_hours: float = 24.0) -> bool:
    """Check if feature extraction should be run based on last run time."""
    last_run = get_last_extraction_run(db)
    
    if last_run is None:
        logger.info("No previous feature extraction run found")
        return True
    
    if last_run.status != "completed":
        logger.info(f"Last run status: {last_run.status}")
        return True
    
    hours_since_last = (Time.now().mjd - last_run.mjd_run) * 24
    
    if hours_since_last > max_age_hours:
        logger.info(f"Last run was {hours_since_last:.1f} hours ago (max: {max_age_hours})")
        return True
    
    logger.info(f"Last run was {hours_since_last:.1f} hours ago, no need to run")
    return False 