"""Feature extraction module for transient objects."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from astropy.time import Time
from astropy import units as u
import logging
import time
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

from . import models
from .filter_manager import filter_manager

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_daily_objects(lookback_days: float = 20.0, lookback_t_first: float = 500, test_mode: bool = False) -> Optional[pd.DataFrame]:
    """
    Get objects with detections in the last N days.
    
    This queries the Antares API for recent objects.
    If Antares is not available, returns None to allow the system to continue.
    
    Args:
        lookback_days: Number of days to look back
        lookback_t_first: Not used in current implementation
        test_mode: If True, skip Antares query entirely and return None gracefully
    """
    logger.info(f"Getting objects with detections in the last {lookback_days} days (test_mode={test_mode})")
    
    if test_mode:
        logger.info("Test mode enabled - skipping Antares query")
        return None
    
    try:
        # Import antares_client here to avoid startup issues if not installed
        try:
            import antares_client
            from antares_client.search import search
            logger.debug(f"Successfully imported antares_client version: {getattr(antares_client, '__version__', 'unknown')}")
        except ImportError as e:
            logger.warning(f"antares_client not installed - feature extraction will skip new object discovery: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error importing antares_client - feature extraction will skip new object discovery: {e}")
            return None
        
        # Calculate time range
        end_time = Time.now()
        start_time = end_time - lookback_days * u.day
        
        # Search for objects with recent detections
        query = {
            'query': {
                'bool': {
                    'must': [
                        {
                            'range': {
                                'properties.newest_alert_observation_time': {
                                    'gte': start_time.mjd,
                                    'lte': end_time.mjd
                                }
                            }
                        },
                        {
                            'range': {
                                'properties.ztf_object_id': {
                                    'gte': 'ZTF18',  # Only ZTF objects
                                    'lte': 'ZTF99'
                                }
                            }
                        }
                    ]
                }
            },
            'sort': [
                {'properties.newest_alert_observation_time': {'order': 'desc'}}
            ]
        }
        
        logger.info(f"Querying Antares for objects detected between MJD {start_time.mjd:.1f} and {end_time.mjd:.1f}")
        logger.debug(f"Query parameters: {query}")
        
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Antares query timed out after 60 seconds")
            
            # Set up timeout for the query
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout
            
            try:
                logger.info("Starting Antares API query (60s timeout)...")
                results = search(query)  # Remove 'limit' parameter - not supported in newer antares_client
                
                # Convert to list first to check if empty
                logger.info("Converting Antares results to list...")
                results_list = list(results) if results else []
                
                logger.info(f"Antares returned {len(results_list)} raw results")
            finally:
                # Cancel the alarm
                signal.alarm(0)
                
        except TimeoutError as e:
            logger.error(f"Antares query timed out: {e}")
            logger.info("Antares server may be slow or unresponsive. Try again later.")
            return None
        except Exception as e:
            logger.error(f"Antares API query failed: {e}")
            logger.info("This is likely a temporary Antares server issue. Try again later.")
            return None
        
        if not results_list:
            logger.warning(f"No objects found in Antares query for the last {lookback_days} days")
            
            # Try a broader search to see if Antares is responsive
            if lookback_days < 30:
                logger.info("Trying broader search to test Antares connectivity...")
                try:
                    broader_query = {
                        'query': {
                            'bool': {
                                'must': [
                                    {
                                        'range': {
                                            'properties.newest_alert_observation_time': {
                                                'gte': (end_time - 30 * u.day).mjd,
                                                'lte': end_time.mjd
                                            }
                                        }
                                    },
                                    {
                                        'range': {
                                            'properties.ztf_object_id': {
                                                'gte': 'ZTF18',
                                                'lte': 'ZTF99'
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                    
                    broader_results = list(search(broader_query))
                    logger.info(f"Broader 30-day search found {len(broader_results)} objects")
                    
                    if len(broader_results) > 0:
                        logger.info("Antares is working, just no recent detections in your timeframe")
                    else:
                        logger.warning("Even 30-day search found nothing - possible Antares issue")
                        
                except Exception as e:
                    logger.warning(f"Broader search failed: {e}")
            
            logger.info(f"This is normal - ZTF doesn't necessarily have new detections every day")
            logger.info(f"Try increasing lookback_days if you expect more objects")
            return None
        
        # Convert to DataFrame with improved debugging
        objects_data = []
        logger.info(f"Processing {len(results_list)} loci from Antares...")
        
        for i, locus in enumerate(results_list):
            try:
                # Enhanced debugging for first few loci
                if i < 3:
                    logger.debug(f"=== Processing locus {i+1} ===")
                    logger.debug(f"Locus type: {type(locus)}")
                    logger.debug(f"Locus attributes: {sorted([attr for attr in dir(locus) if not attr.startswith('_')])}")
                    
                    if hasattr(locus, 'properties'):
                        logger.debug(f"Properties: {locus.properties}")
                    if hasattr(locus, 'catalog_objects'):
                        logger.debug(f"Catalog objects type: {type(locus.catalog_objects)}")
                        if locus.catalog_objects:
                            logger.debug(f"Catalog objects content: {locus.catalog_objects}")
                
                # Try different ways to extract ZTFID - enhanced
                ztfid = None
                
                # Method 1: From catalog_objects
                if hasattr(locus, 'catalog_objects') and locus.catalog_objects:
                    if isinstance(locus.catalog_objects, list) and len(locus.catalog_objects) > 0:
                        catalog_obj = locus.catalog_objects[0]
                        if hasattr(catalog_obj, 'catalog_object_id'):
                            ztfid = catalog_obj.catalog_object_id
                        elif isinstance(catalog_obj, dict):
                            ztfid = catalog_obj.get('catalog_object_id')
                    elif isinstance(locus.catalog_objects, dict):
                        ztfid = locus.catalog_objects.get('catalog_object_id')
                
                # Method 2: From locus properties
                if not ztfid and hasattr(locus, 'properties') and locus.properties:
                    # Try multiple property names
                    for prop_name in ['ztf_object_id', 'ztfid', 'object_id', 'name']:
                        if prop_name in locus.properties:
                            candidate_id = locus.properties[prop_name]
                            if candidate_id and str(candidate_id).startswith('ZTF'):
                                ztfid = candidate_id
                                break
                
                # Method 3: Direct locus fields
                if not ztfid:
                    for attr_name in ['locus_id', 'object_id', 'name']:
                        if hasattr(locus, attr_name):
                            candidate_id = getattr(locus, attr_name)
                            if candidate_id and str(candidate_id).startswith('ZTF'):
                                ztfid = candidate_id
                                break
                
                # Method 4: Check if there's a to_dict method
                if not ztfid and hasattr(locus, 'to_dict'):
                    try:
                        locus_dict = locus.to_dict()
                        for key in ['ztf_object_id', 'object_id', 'name', 'ztfid']:
                            if key in locus_dict and str(locus_dict[key]).startswith('ZTF'):
                                ztfid = locus_dict[key]
                                break
                    except Exception as e:
                        logger.debug(f"Error calling to_dict(): {e}")
                
                if not ztfid:
                    if i < 5:  # Only warn for first few failures
                        logger.warning(f"Could not extract ZTFID from locus {i+1}")
                        logger.debug(f"Available properties: {list(locus.properties.keys()) if hasattr(locus, 'properties') and locus.properties else 'No properties'}")
                    continue
                
                # Ensure ZTFID looks like a ZTF ID
                if not str(ztfid).startswith('ZTF'):
                    if i < 5:
                        logger.warning(f"Invalid ZTFID format: {ztfid}")
                    continue
                
                # Enhanced coordinate extraction
                ra = dec = None
                
                # Try direct attributes first
                if hasattr(locus, 'ra'):
                    ra = locus.ra
                if hasattr(locus, 'dec'):
                    dec = locus.dec
                
                # Try properties if direct attributes didn't work
                if (ra is None or dec is None) and hasattr(locus, 'properties') and locus.properties:
                    ra = ra or locus.properties.get('ra')
                    dec = dec or locus.properties.get('dec')
                
                if ra is None or dec is None:
                    if i < 5:
                        logger.warning(f"Missing coordinates for {ztfid}: ra={ra}, dec={dec}")
                    continue
                
                # Get newest alert time
                newest_alert = 0
                if hasattr(locus, 'properties') and locus.properties:
                    newest_alert = locus.properties.get('newest_alert_observation_time', 0)
                
                objects_data.append({
                    'ZTFID': str(ztfid),
                    'ra': float(ra),
                    'dec': float(dec),
                    'newest_alert': float(newest_alert),
                    'locus': locus  # Store the full locus for later processing
                })
                
                if i < 5:
                    logger.debug(f"Successfully processed {ztfid} at RA={ra:.4f}, Dec={dec:.4f}")
                
            except Exception as e:
                if i < 10:  # Only log errors for first 10 failures
                    logger.warning(f"Error processing locus {i+1}: {e}")
                    logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
                continue
        
        if not objects_data:
            logger.warning("No valid objects extracted from Antares results")
            return None
        
        df = pd.DataFrame(objects_data)
        logger.info(f"Found {len(df)} objects with recent detections")
        return df
        
    except Exception as e:
        logger.error(f"Error querying Antares: {e}")
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
    
    This fetches the light curve from Antares and extracts features.
    """
    ztfid = obj_row.get('ZTFID', 'unknown')
    
    try:
        # Get the locus from the object row
        locus = obj_row.get('locus')
        if locus is None:
            # Try to fetch from Antares by ZTFID
            try:
                from antares_client.search import get_by_ztf_object_id
                locus = get_by_ztf_object_id(ztfid)
            except ImportError:
                logger.error(f"antares_client not available for {ztfid}")
                return None
        
        if locus is None:
            logger.warning(f"Could not fetch locus for {ztfid}")
            return None
        
        # Get timeseries data
        timeseries_data = []
        for alert in locus.alerts:
            for obs in alert.observations:
                if hasattr(obs, 'mag') and hasattr(obs, 'mjd') and hasattr(obs, 'passband'):
                    # Only include valid detections
                    if not np.isnan(obs.mag) and obs.mag > 0:
                        timeseries_data.append({
                            'ant_mjd': obs.mjd,
                            'ant_mag': obs.mag,
                            'ant_magerr': getattr(obs, 'magerr', 0.1),
                            'ant_passband': obs.passband
                        })
        
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
        features['ra'] = obj_row.get('ra', locus.ra)
        features['dec'] = obj_row.get('dec', locus.dec)
        
        # Create feature errors (simplified - use 10% of feature value or default)
        feature_errors = {}
        for key, value in features.items():
            if key in ['ZTFID', 'ra', 'dec']:
                continue
            if np.isnan(value) or value == 0:
                feature_errors[f'{key}_err'] = np.nan
            else:
                feature_errors[f'{key}_err'] = abs(value) * 0.1  # 10% error
        
        # Combine features and errors
        all_features = {**features, **feature_errors}
        
        # Convert to DataFrame
        result_df = pd.DataFrame([all_features])
        
        logger.info(f"Successfully processed {ztfid}")
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
                logger.warning("No new objects found from Antares query")
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