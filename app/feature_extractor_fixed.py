"""Fixed version of get_daily_objects that works in threaded environments."""

import numpy as np
import pandas as pd
from typing import Optional
from astropy.time import Time
from astropy import units as u
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime

# Optional imports with error handling
try:
    import antares_client
    from antares_client.search import search, get_by_ztf_object_id
except ImportError:
    antares_client = None
    search = None
    get_by_ztf_object_id = None

logger = logging.getLogger(__name__)

def execute_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """Execute a function with a timeout using threading instead of signals."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            # The thread will continue running but we abandon waiting for it
            raise TimeoutError(f"Function execution timed out after {timeout_seconds} seconds")

def get_daily_objects_fixed(lookback_days: float = 20.0, lookback_t_first: float = 500, test_mode: bool = False) -> Optional[pd.DataFrame]:
    """
    Get objects with detections in the last N days.
    Fixed version that works in threaded environments (no signal handling).
    """
    logger.info(f"Getting objects with detections in the last {lookback_days} days (test_mode={test_mode})")
    
    if test_mode:
        logger.info("Test mode enabled - skipping Antares query")
        return None
    
    try:
        # Check if antares_client is available 
        if antares_client is None or search is None:
            logger.warning("antares_client not installed - feature extraction will skip new object discovery")
            return None
        
        # Test ANTARES API responsiveness with a simple query
        logger.info("Testing ANTARES API responsiveness...")
        try:
            def test_api():
                test_query = {
                    'query': {'match_all': {}},
                    'size': 1
                }
                results = search(test_query)
                return list(results)
            
            # Use threading-based timeout instead of signal
            test_results = execute_with_timeout(test_api, timeout_seconds=15)
            logger.info(f"✓ ANTARES API test successful - API is responsive")
            
        except TimeoutError:
            logger.error("✗ ANTARES API is unresponsive (timed out in 15 seconds)")
            logger.info("Feature extraction will continue without new object discovery")
            return None
        except Exception as e:
            logger.error(f"✗ ANTARES API test failed: {e}")
            logger.info("Feature extraction will continue without new object discovery")
            return None
        
        # Calculate time range
        end_time = Time.now()
        start_time = end_time - lookback_days * u.day
        
        logger.info(f"Querying Antares for objects detected between MJD {start_time.mjd:.1f} and {end_time.mjd:.1f}")
        
        total_results = []
        
        # Use daily chunks to handle large queries
        time_chunk_days = 1.0
        max_time_chunks = int(np.ceil(lookback_days / time_chunk_days))
        
        logger.info(f"Splitting {lookback_days} days into {max_time_chunks} chunks")
        
        for chunk_idx in range(max_time_chunks):
            chunk_start_time = end_time - (chunk_idx + 1) * time_chunk_days * u.day
            chunk_end_time = end_time - chunk_idx * time_chunk_days * u.day
            
            if chunk_start_time.mjd < start_time.mjd:
                chunk_start_time = start_time
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{max_time_chunks}: MJD {chunk_start_time.mjd:.2f} to {chunk_end_time.mjd:.2f}")
            
            # Try different batch sizes
            batch_sizes = [50, 20, 10, 5]
            chunk_success = False
            
            for batch_size in batch_sizes:
                if chunk_success:
                    break
                
                try:
                    def execute_chunk_query():
                        chunk_query = {
                            'query': {
                                'bool': {
                                    'must': [
                                        {
                                            'range': {
                                                'properties.newest_alert_observation_time': {
                                                    'gte': chunk_start_time.mjd,
                                                    'lte': chunk_end_time.mjd
                                                }
                                            }
                                        }
                                    ]
                                }
                            },
                            'sort': [
                                {'properties.newest_alert_observation_time': {'order': 'desc'}}
                            ],
                            'size': batch_size
                        }
                        
                        results = search(chunk_query)
                        return list(results) if results else []
                    
                    # Use threading timeout
                    timeout = min(30, batch_size * 2)  # Scale timeout with batch size
                    logger.info(f"  Trying batch size {batch_size} with {timeout}s timeout")
                    
                    chunk_results = execute_with_timeout(execute_chunk_query, timeout_seconds=timeout)
                    
                    if chunk_results:
                        total_results.extend(chunk_results)
                        chunk_success = True
                        logger.info(f"  ✓ Retrieved {len(chunk_results)} objects")
                        break
                    else:
                        logger.info(f"  No results with batch size {batch_size}")
                        
                except TimeoutError:
                    logger.warning(f"  Timeout with batch size {batch_size}")
                    continue
                except Exception as e:
                    logger.warning(f"  Error with batch size {batch_size}: {e}")
                    continue
            
            if not chunk_success:
                logger.warning(f"Failed to retrieve objects for chunk {chunk_idx + 1}")
            
            # Small delay between chunks
            if chunk_idx < max_time_chunks - 1:
                time.sleep(0.5)
        
        # Remove duplicates
        seen_ids = set()
        deduplicated_results = []
        
        for obj in total_results:
            obj_id = None
            
            # Extract ZTFID
            if hasattr(obj, 'catalog_objects') and obj.catalog_objects:
                if isinstance(obj.catalog_objects, list) and len(obj.catalog_objects) > 0:
                    catalog_obj = obj.catalog_objects[0]
                    if hasattr(catalog_obj, 'catalog_object_id'):
                        obj_id = catalog_obj.catalog_object_id
            
            if obj_id and obj_id not in seen_ids:
                seen_ids.add(obj_id)
                deduplicated_results.append(obj)
            elif not obj_id:
                deduplicated_results.append(obj)
        
        logger.info(f"Query complete: found {len(total_results)} total objects, {len(deduplicated_results)} unique")
        
        if not deduplicated_results:
            logger.warning("No objects found in time range")
            
            # Test individual object query as sanity check
            try:
                def test_individual():
                    return get_by_ztf_object_id('ZTF18aabtxvd')
                
                test_obj = execute_with_timeout(test_individual, timeout_seconds=10)
                if test_obj:
                    logger.info("✓ Individual object queries work")
                    logger.warning("✗ But time-based bulk queries returned no results")
            except Exception as e:
                logger.error(f"✗ Even individual queries failing: {e}")
            
            return None
        
        # Convert to DataFrame
        objects_data = []
        
        for locus in deduplicated_results:
            try:
                # Extract ZTFID
                ztfid = None
                
                if hasattr(locus, 'catalog_objects') and locus.catalog_objects:
                    if isinstance(locus.catalog_objects, list) and len(locus.catalog_objects) > 0:
                        catalog_obj = locus.catalog_objects[0]
                        if hasattr(catalog_obj, 'catalog_object_id'):
                            ztfid = catalog_obj.catalog_object_id
                
                if not ztfid and hasattr(locus, 'properties') and locus.properties:
                    for prop_name in ['ztf_object_id', 'ztfid', 'object_id', 'name']:
                        if prop_name in locus.properties:
                            candidate_id = locus.properties[prop_name]
                            if candidate_id and str(candidate_id).startswith('ZTF'):
                                ztfid = candidate_id
                                break
                
                if not ztfid or not str(ztfid).startswith('ZTF'):
                    continue
                
                # Extract coordinates
                ra = dec = None
                
                if hasattr(locus, 'ra'):
                    ra = locus.ra
                if hasattr(locus, 'dec'):
                    dec = locus.dec
                
                if (ra is None or dec is None) and hasattr(locus, 'properties') and locus.properties:
                    ra = ra or locus.properties.get('ra')
                    dec = dec or locus.properties.get('dec')
                
                if ra is None or dec is None:
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
                    'locus': locus
                })
                
            except Exception as e:
                logger.debug(f"Error processing locus: {e}")
                continue
        
        if not objects_data:
            logger.warning("No valid objects extracted from results")
            return None
        
        df = pd.DataFrame(objects_data)
        logger.info(f"Extracted {len(df)} valid objects with coordinates")
        return df
        
    except Exception as e:
        logger.error(f"Error in get_daily_objects_fixed: {e}")
        return None 