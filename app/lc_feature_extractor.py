import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import find_peaks
import antares_client
from extinction import apply as apply_extinction
import sfdmap
from astropy.time import Time
import matplotlib.pyplot as plt
import seaborn as sns
from dust_extinction.parameter_averages import G23
from numpy.lib.stride_tricks import sliding_window_view
import time
from sklearn.cluster import DBSCAN
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import requests
import warnings
from tqdm import tqdm
from scipy.spatial import distance
from scipy.stats import gamma, halfnorm, uniform
from astropy.coordinates import SkyCoord
from astropy import units as u
from astro_prost.helpers import SnRateAbsmag
from astro_prost.associate import associate_sample

warnings.filterwarnings("ignore", category=RuntimeWarning)

VAR_CATALOGS = [
    "gaia_dr3_variability",
    "sdss_stars",
    "bright_guide_star_cat",
    "asassn_variable_catalog_v2_20190802",
    "vsx",
    "linear_ll",
    "veron_agn_qso", # agn/qso
    "milliquas", # qso
]


ZP = 27.5
SNT = 3


def get_daily_objects(lookback_t=5, lookback_t_first=1000, savepath=None):
    if (savepath is not None) and (not os.path.exists(savepath)):
        os.makedirs(savepath)

    today_mjd = Time.now().mjd

    loci = antares_client.search.search(
        {
            "query": {
                "bool": {
                    "must":[
                         {"range":{"properties.num_mag_values":{"gte": 5}}},
                         {"range":{"properties.newest_alert_observation_time": {"gte": today_mjd - lookback_t}}},
                         {"range":{"properties.brightest_alert_magnitude":{"lte": "21"}}},
                         {"range":{"properties.oldest_alert_observation_time": {"gte": today_mjd - lookback_t_first}}}
                         ],
                    "must_not": [  # Exclude loci that contain any of these catalogs
                        {"terms": {"locus.catalogs": VAR_CATALOGS}}]
                        }
                    }
        }
    )

    loci_list = list(loci)
    print(f"Retrieved {len(loci_list)} loci")
    if len(loci_list) < 1:
        return None

    # Use 8 workers or fewer if < 8 loci
    num_workers = min(8, len(loci_list))

    with Pool(num_workers) as pool:
        results = pool.map(process_locus, loci_list)

    #get just the metadata for now
    transient_cat_list = [r for r in results]
    transient_catalog = pd.concat(transient_cat_list, ignore_index=True)

    # define priors for properties
    priorfunc_offset = uniform(loc=0, scale=10)
    likefunc_offset = gamma(a=0.75)

    catalogs = ["glade", "decals", "panstarrs"]
    transient_coord_cols = ("ra", "dec")
    transient_name_col = "name"
    verbose = 0

    priors = {"offset": priorfunc_offset}
    likes = {"offset": likefunc_offset}

    transient_catalog_with_hosts = \
        associate_sample(
            transient_catalog,
            priors=priors,
            likes=likes,
            catalogs=catalogs,
            parallel=True,
            verbose=verbose,
            save=False,
            name_col=transient_name_col,
            coord_cols=transient_coord_cols,
            progress_bar=True,
            cat_cols=False)

    # if TNS redshift, replace host redshift
    tns_redshifts = transient_catalog_with_hosts['tns_redshift'].values
    transient_catalog_with_hosts.loc[tns_redshifts == tns_redshifts, 'host_redshift_mean'] = tns_redshifts[tns_redshifts == tns_redshifts]
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    transient_catalog_with_hosts['peak_mag'] -= cosmo.distmod(transient_catalog_with_hosts['host_redshift_mean'].values).value

    offset_bool = transient_catalog_with_hosts['host_offset_mean'] >= 2. # 2" offset to eliminate AGN/flaring stars

    daily_transients = transient_catalog_with_hosts[offset_bool]

    if savepath is not None:
        daily_transients.to_csv(savepath + "/dailytransients_{int(today_mjd)}.csv")
    return daily_transients

def process_locus(locus):
    """Process a single locus ID: fetch forced photometry, save CSV, return transient info"""
    today_mjd = Time.now().mjd
    locus_id = locus.locus_id
    discovery_date = Time(locus.properties['oldest_alert_observation_time'], format='mjd')
    ZTFID = locus.properties.get('ztf_object_id')
    oldest_alert =  locus.properties['oldest_alert_observation_time']
    newest_alert = locus.properties['newest_alert_observation_time']
    latest_alert_mag = locus.properties['newest_alert_magnitude']
    brightest_alert_mag = locus.properties['brightest_alert_magnitude']
    peak_phase = today_mjd - locus.properties['brightest_alert_observation_time']
    duration = newest_alert - oldest_alert

    if 'tns_public_objects' in locus.catalog_objects:
        tns = locus.catalog_objects['tns_public_objects'][0]
        tns_name, tns_cls, tns_redshift = tns['name'], tns['type'], tns['redshift']
        if tns_cls == '':
            tns_cls = '---'
        if tns_redshift is None:
            tns_redshift = np.nan
    else:
        tns_name = '---'
        tns_cls = '---'
        tns_redshift = np.nan

    df = pd.DataFrame({'name':locus_id, 'ZTFID':ZTFID, 'ra':locus.coordinates.ra.deg, 'dec':locus.coordinates.dec.deg,\
        'oldest_alert':oldest_alert, 'newest_alert':newest_alert, 'latest_magnitude':latest_alert_mag, 'peak_mag':brightest_alert_mag,
        'tns_name':tns_name, 'tns_class':tns_cls, 'duration':duration, 'tns_redshift':tns_redshift, 'peak_phase':peak_phase}, index=[0])

    return df

def fetch_timeseries(ztf_id):
    """Fetch the timeseries for a given ZTF ID, with error handling."""
    try:
        obj = antares_client.search.get_by_ztf_object_id(ztf_id)
        # Check if timeseries is available
        if hasattr(obj, 'timeseries') and obj.timeseries is not None:
            df = obj.timeseries.to_pandas()
            return ztf_id, df
        else:
            #print(f"[Warning] No timeseries data for {ztf_id}")
            return ztf_id, None
    except requests.exceptions.ConnectionError as e:
        print(f"---- [Error] Fetching timeseries for {ztf_id} failed: {e}")
        time.sleep(2)
        print("---- Slept, trying again...")
        obj = antares_client.search.get_by_ztf_object_id(ztf_id)
        # Check if timeseries is available
        if hasattr(obj, 'timeseries') and obj.timeseries is not None:
            df = obj.timeseries.to_pandas()
            return ztf_id, df
        else:
            return ztf_id, None

def process_object(row, return_uncertainty=True):
    ZTFID = row['ZTFID']
    try:
        ZTFID, df = fetch_timeseries(ZTFID)
        if df is None:
            return None
        df = df[df['ant_mjd'].notna() & df['ant_mag'].notna() &
                df['ant_magerr'].notna() & df['ant_passband'].notna()]
        if (np.nansum(df['ant_passband'] == 'R') < 2 or
            np.nansum(df['ant_passband'] == 'g') < 2):
            return None

        time_g = df.loc[df['ant_passband'] == 'g', 'ant_mjd'].values
        mag_g = df.loc[df['ant_passband'] == 'g', 'ant_mag'].values
        err_g = df.loc[df['ant_passband'] == 'g', 'ant_magerr'].values
        time_r = df.loc[df['ant_passband'] == 'R', 'ant_mjd'].values
        mag_r = df.loc[df['ant_passband'] == 'R', 'ant_mag'].values
        err_r = df.loc[df['ant_passband'] == 'R', 'ant_magerr'].values

        extractor = SupernovaFeatureExtractor(ZTFID, time_g, mag_g, err_g, time_r, mag_r, err_r)
        features = extractor.extract_features(return_uncertainty=return_uncertainty)
        if features is None:
            return None
        features['ZTFID'] = row['ZTFID']

    except Exception as e:
        print(f"---- [Error] Failed on {ZTFID}: {e}")
        return None

    #tack on locus features
    for l_feature in ['tns_redshift', 'latest_magnitude', 'oldest_alert', 'newest_alert', 'peak_phase', 'latest_magnitude']:
        try:
            features[l_feature] = row[l_feature]
        except:
            features[l_feature] = row[l_feature + "_x"]
    return features

def local_curvature(times, mags):
    if len(times) < 3:
        return np.nan
    curvatures = []
    for i in range(1, len(times) - 1):
        t0, t1, t2 = times[i - 1], times[i], times[i + 1]
        m0, m1, m2 = mags[i - 1], mags[i], mags[i + 1]
        dt = t2 - t0
        if dt == 0:
            continue
        a = (m2 - 2 * m1 + m0) / ((dt / 2) ** 2)
        curvatures.append(a)
    return np.median(curvatures) if curvatures else np.nan

def pairwise_chisq_distances(query, query_err, X, X_err):
    query = np.asarray(query)
    query_err = np.asarray(query_err)
    X = np.asarray(X)
    X_err = np.asarray(X_err)

    # Only keep dimensions where all values are finite
    valid_mask = np.isfinite(query) & np.isfinite(query_err)
    valid_mask &= np.isfinite(X).all(axis=0) & np.isfinite(X_err).all(axis=0)

    if np.sum(valid_mask) < 2:
        return np.full(X.shape[0], np.inf)

    # Subset valid dimensions
    q = query[valid_mask]
    q_err = query_err[valid_mask]
    X_valid = X[:, valid_mask]
    X_err_valid = X_err[:, valid_mask]

    # Combine query and databank variances elementwise
    # ignore databank variances for now
    #combined_var = q_err**2 + X_err_valid**2

    # compute distance
    chi2 = np.sum(((X_valid - q) ** 2) / q_err**2, axis=1)

    return np.sqrt(chi2)

def local_curvature(times, mags):
    if len(times) < 3:
        return np.nan
    curvatures = []
    for i in range(1, len(times) - 1):
        t0, t1, t2 = times[i - 1], times[i], times[i + 1]
        m0, m1, m2 = mags[i - 1], mags[i], mags[i + 1]
        dt = t2 - t0
        if dt == 0:
            continue
        a = (m2 - 2 * m1 + m0) / ((dt / 2) ** 2)
        curvatures.append(a)
    return np.median(curvatures) if curvatures else np.nan

m = sfdmap.SFDMap()

class SupernovaFeatureExtractor:
    @staticmethod
    def describe_features():
        return {
            't0': 'Time zero-point for light curve normalization',
            'g_peak_mag': 'Minimum magnitude (brightest point) in g band',
            'g_peak_time': 'Time of peak brightness in g band',
            'g_rise_time': 'Time from 50% peak flux to g-band peak',
            'g_decline_time': 'Time from g-band peak to 50% flux decay',
            'g_duration_above_half_flux': 'Duration above 50% of g-band peak flux',
            'r_peak_mag': 'Minimum magnitude (brightest point) in r band',
            'r_peak_time': 'Time of peak brightness in r band',
            'r_rise_time': 'Time from 50% peak flux to r-band peak',
            'r_decline_time': 'Time from r-band peak to 50% flux decay',
            'r_duration_above_half_flux': 'Duration above 50% of r-band peak flux',
            'g_amplitude': 'Magnitude difference between min and max in g band',
            'g_skewness': 'Skew of g-band magnitude distribution',
            'g_beyond_2sigma': 'Fraction of g-band points beyond 2 standard deviations',
            'r_amplitude': 'Magnitude difference between min and max in r band',
            'r_skewness': 'Skew of r-band magnitude distribution',
            'r_beyond_2sigma': 'Fraction of r-band points beyond 2 standard deviations',
            'g_max_rolling_variance': 'Max variance in a 5-point window in g band',
            'g_mean_rolling_variance': 'Mean variance in a 5-point window in g band',
            'r_max_rolling_variance': 'Max variance in a 5-point window in r band',
            'r_mean_rolling_variance': 'Mean variance in a 5-point window in r band',
            'mean_g-r': 'Average g-r color over shared time range',
            'g-r_at_g_peak': 'g-r color at g-band peak time',
            'mean_color_rate': 'Average rate of change of g-r color',
            'g_n_peaks': 'Number of peaks in g band (prominence > 0.1)',
            'g_dt_main_to_secondary_peak': 'Time difference between top two g-band peaks',
            'g_dmag_secondary_peak': 'Magnitude difference between g-band peaks',
            'g_secondary_peak_prominence': 'Prominence of second-brightest g-band peak',
            'g_secondary_peak_width': 'Width of second-brightest g-band peak',
            'r_n_peaks': 'Number of peaks in r band (prominence > 0.1)',
            'r_dt_main_to_secondary_peak': 'Time difference between top two r-band peaks',
            'r_dmag_secondary_peak': 'Magnitude difference between r-band peaks',
            'r_secondary_peak_prominence': 'Prominence of second-brightest r-band peak',
            'r_secondary_peak_width': 'Width of second-brightest r-band peak',
            'g_rise_local_curvature': 'Median second derivative of rising g-band light curve (20d window)',
            'g_decline_local_curvature': 'Median second derivative of declining g-band light curve (20d window)',
            'r_rise_local_curvature': 'Median second derivative of rising r-band light curve (20d window)',
            'r_decline_local_curvature': 'Median second derivative of declining r-band light curve (20d window)',
            'features_valid': 'Whether all key features were successfully computed',
            'ZTFID': 'ZTF object identifier'
        }

    def __init__(self, ZTFID, time_g, mag_g, err_g, time_r, mag_r, err_r, ra=None, dec=None):
        self.ZTFID = ZTFID
        self.g = {'time': np.array(time_g), 'mag': np.array(mag_g), 'err': np.array(err_g)}
        self.r = {'time': np.array(time_r), 'mag': np.array(mag_r), 'err': np.array(err_r)}
        t0 = min(self.g['time'].min(), self.r['time'].min())
        self.time_offset = t0
        self.g['time'] -= t0
        self.r['time'] -= t0
        if ra is not None and dec is not None:
            ebv = m.ebv(ra, dec)
            ext = G23(Rv=3.1)
            lambda_g = 0.477
            lambda_r = 0.623
            Ag = ext.extinguish(lambda_g, Ebv=ebv, unit='um')
            Ar = ext.extinguish(lambda_r, Ebv=ebv, unit='um')
            self.g['mag'] -= -2.5 * np.log10(Ag)
            self.r['mag'] -= -2.5 * np.log10(Ar)
        self._preprocess()

    def _preprocess(self, min_cluster_size=2):
        for band_name in ['g', 'r']:
            band = getattr(self, band_name)
            idx = np.argsort(band['time'])
            for key in band:
                band[key] = band[key][idx]

            time_reshaped = band['time'].reshape(-1, 1)
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=5, min_samples=min_cluster_size).fit(time_reshaped)
            labels = clustering.labels_

            # Keep all points that are part of clusters with ≥ min_cluster_size
            good_clusters = [
                label for label in set(labels) if label != -1 and np.sum(labels == label) >= min_cluster_size
            ]
            mask = np.isin(labels, good_clusters)

            for key in band:
                band[key] = band[key][mask]

        # Recalculate t0 based on filtered times
        if len(self.g['time']) == 0 or len(self.r['time']) == 0:
            print(f"[Warning] No data left in g or r band after filtering for object {self.ZTFID}. Skipping.")
        else:
            new_time_offset = min(self.g['time'].min(), self.r['time'].min())

            # Normalize times again
            self.g['time'] -= new_time_offset
            self.r['time'] -= new_time_offset

            self.time_offset += new_time_offset

    def _select_main_cluster(self, time, mag, min_samples=3, eps=20):
        from sklearn.cluster import DBSCAN
        if len(time) < min_samples:
            return np.ones_like(time, dtype=bool)
        time_reshaped = np.array(time).reshape(-1, 1)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(time_reshaped)
        labels = clustering.labels_
        if all(labels == -1):
            return np.ones_like(time, dtype=bool)
        unique_labels = np.unique(labels[labels != -1])
        best_label = None
        best_brightness = np.inf
        for label in unique_labels:
            mask = labels == label
            cluster_time = np.array(time)[mask]
            cluster_span = cluster_time.max() - cluster_time.min()
            cluster_mag = np.min(np.array(mag)[mask])
            # Prioritize clusters with brighter peak and tighter time span
            score = cluster_mag + 0.05 * cluster_span
            if score < best_brightness:
                best_brightness = score
                best_label = label
        return labels == best_label

    def _flag_isolated_points(time, max_gap_factor=5):
        time = np.sort(time)
        dt = np.diff(time)

        # Median cadence (ignoring gaps)
        median_dt = np.median(dt)

        # Find large gaps
        gaps = np.concatenate([[0], dt > max_gap_factor * median_dt])

        # Mark isolated points as True
        isolated = np.zeros_like(time, dtype=bool)
        for i in range(1, len(time) - 1):
            if gaps[i] and gaps[i+1]:
                isolated[i] = True
        return isolated

    def _core_stats(self, band):
        t, m = band['time'], band['mag']
        mask = np.isfinite(t) & np.isfinite(m) & ~np.isnan(m)
        t, m = t[mask], m[mask]
        t, idx = np.unique(t, return_index=True)
        m = m[idx]
        if len(m) < 3 or np.ptp(m) < 0.2:
            #print("Warning: Not enough valid data or insufficient variability to compute core stats.")
            return np.nan, np.nan, np.nan, np.nan, np.nan
        peak_idx = np.argmin(m)
        peak_mag = m[peak_idx]
        peak_time = t[peak_idx]
        flux = 10**(-0.4 * m)
        half_flux = 0.5 * 10**(-0.4 * peak_mag)
        half_mag = -2.5 * np.log10(half_flux)
        pre, post = t < peak_time, t > peak_time
        try:
            rise_time = peak_time - np.interp(half_mag, m[pre][::-1], t[pre][::-1])
        except:
            rise_time = np.nan
        try:
            decline_time = np.interp(half_mag, m[post], t[post]) - peak_time
        except:
            decline_time = np.nan
        above_half = t[m < half_mag]
        duration = above_half[-1] - above_half[0] if len(above_half) > 1 else np.nan
        return peak_mag, peak_time, rise_time, decline_time, duration

    def _variability_stats(self, band):
        mag = band['mag']
        amp = np.max(mag) - np.min(mag)
        std = np.std(mag)
        mean = np.mean(mag)
        skew = (np.mean((mag - mean)**3) / std**3) if std > 0 else np.nan
        beyond_2 = np.sum(np.abs(mag - mean) > 2 * std) / len(mag)
        return amp, skew, beyond_2

    def _color_features(self):
        if len(self.g['time']) < 2 or len(self.r['time']) < 2:
            print("Warning: Not enough data in g or r band to compute color features.")
            return None
        def dedup(t, m):
            mask = np.isfinite(t) & np.isfinite(m) & ~np.isnan(m)
            t, m = t[mask], m[mask]
            _, idx = np.unique(t, return_index=True)
            return t[idx], m[idx]

        t_min = max(self.g['time'].min(), self.r['time'].min())
        t_max = min(self.g['time'].max(), self.r['time'].max())

        if t_max <= t_min or np.isnan(t_min) or np.isnan(t_max):
            #print("Warning: No overlapping time range for g and r bands.")
            return None
        t_grid = np.linspace(t_min, t_max, 100)
        g_time, g_mag = dedup(self.g['time'], self.g['mag'])
        r_time, r_mag = dedup(self.r['time'], self.r['mag'])
        g_interp = interp1d(g_time, g_mag, kind='linear', fill_value='extrapolate')
        r_interp = interp1d(r_time, r_mag, kind='linear', fill_value='extrapolate')
        color = g_interp(t_grid) - r_interp(t_grid)

        dcolor_dt = np.gradient(color, t_grid)
        mean_rate = np.mean(dcolor_dt)
        tpg = self.g['time'][np.argmin(self.g['mag'])]
        try:
            gr_at_gpeak = self.g['mag'][np.argmin(self.g['mag'])] - np.interp(tpg, r_time, r_mag)
        except Exception as e:
            print(f"Warning: Could not compute g-r at g-band peak due to: {e}")
            gr_at_gpeak = np.nan
        return np.mean(color), gr_at_gpeak, mean_rate

    def _rolling_variance(self, band, window_size=5):
        def dedup(t, m):
            _, idx = np.unique(t, return_index=True)
            return t[idx], m[idx]
        t_dedup, m_dedup = dedup(band['time'], band['mag'])
        t_uniform = np.linspace(t_dedup.min(), t_dedup.max(), 100)
        mag_interp = interp1d(t_dedup, m_dedup, kind='linear', fill_value='extrapolate')(t_uniform)
        views = sliding_window_view(mag_interp, window_shape=window_size)
        rolling_vars = np.var(views, axis=1)
        return np.max(rolling_vars), np.mean(rolling_vars)

    def _peak_structure(self, band):
        if np.ptp(band['mag']) < 0.5:
            #print("Warning: Insufficient variability to identify peak structure.")
            return 0, np.nan, np.nan, np.nan, np.nan
        t_uniform = np.linspace(band['time'].min(), band['time'].max(), 300)
        mag_interp = interp1d(band['time'], band['mag'], kind='linear', fill_value='extrapolate')(t_uniform)
        peaks, properties = find_peaks(-mag_interp, prominence=0.1, width=5)
        n_peaks = len(peaks)
        if n_peaks < 2:
            return n_peaks, np.nan, np.nan, np.nan, np.nan
        mags = mag_interp[peaks]
        times = t_uniform[peaks]
        prominences = properties['prominences']
        widths = properties['widths']
        main_idx = np.argmin(mags)
        other_idx = np.argsort(mags)[1]
        dt = np.abs(times[main_idx] - times[other_idx])
        dmag = mags[other_idx] - mags[main_idx]
        prominence_second = prominences[other_idx]
        width_second = widths[other_idx]
        return n_peaks, dt, dmag, prominence_second, width_second

    def _local_curvature_features(self, band, window_days=20):
        t, m = band['time'], band['mag']
        mask = np.isfinite(t) & np.isfinite(m)
        t, m = t[mask], m[mask]
        if len(t) < 3:
            return np.nan, np.nan

        # Sort and deduplicate
        t, idx = np.unique(t, return_index=True)
        m = m[idx]

        # Identify peak time
        peak_idx = np.argmin(m)
        t_peak = t[peak_idx]

        # Define ±window around peak
        tmin = t_peak - window_days
        tmax = t_peak + window_days
        local_mask = (t >= tmin) & (t <= tmax)
        t_local, m_local = t[local_mask], m[local_mask]
        if len(t_local) < 3:
            return np.nan, np.nan

        # Split into rise and decline
        rise_t, rise_m = t_local[t_local <= t_peak], m_local[t_local <= t_peak]
        decline_t, decline_m = t_local[t_local >= t_peak], m_local[t_local >= t_peak]

        rise_curv = local_curvature(rise_t, rise_m)
        decline_curv = local_curvature(decline_t, decline_m)
        return rise_curv, decline_curv


    def extract_features(self, return_uncertainty=False, n_trials=20):
        if len(self.g['time']) == 0 or len(self.r['time']) == 0:
            #print(f"[Warning] No data left in g or r band after filtering for object {self.ZTFID}. Skipping.")
            return None

        g_core = self._core_stats(self.g)
        r_core = self._core_stats(self.r)

        g_var = self._variability_stats(self.g)
        r_var = self._variability_stats(self.r)

        color_feats = self._color_features()

        g_rise_curv, g_decline_curv = self._local_curvature_features(self.g)
        r_rise_curv, r_decline_curv = self._local_curvature_features(self.r)

        if color_feats is None:
            #print("Warning: Color features could not be computed. Defaulting to NaN.")
            color_feats = (np.nan, np.nan, np.nan)
        g_peak_struct = self._peak_structure(self.g)
        r_peak_struct = self._peak_structure(self.r)
        g_rollvar = self._rolling_variance(self.g)
        r_rollvar = self._rolling_variance(self.r)
        base_row = {
            't0': self.time_offset,
            'g_peak_mag': g_core[0], 'g_peak_time': g_core[1], 'g_rise_time': g_core[2],
            'g_decline_time': g_core[3], 'g_duration_above_half_flux': g_core[4],
            'g_amplitude': g_var[0], 'g_skewness': g_var[1], 'g_beyond_2sigma': g_var[2],
            'r_peak_mag': r_core[0], 'r_peak_time': r_core[1], 'r_rise_time': r_core[2],
            'r_decline_time': r_core[3], 'r_duration_above_half_flux': r_core[4],
            'r_amplitude': r_var[0], 'r_skewness': r_var[1], 'r_beyond_2sigma': r_var[2],
            'mean_g-r': color_feats[0], 'g-r_at_g_peak': color_feats[1], 'mean_color_rate': color_feats[2],
            'g_n_peaks': g_peak_struct[0], 'g_dt_main_to_secondary_peak': g_peak_struct[1],
            'g_dmag_secondary_peak': g_peak_struct[2], 'g_secondary_peak_prominence': g_peak_struct[3],
            'g_secondary_peak_width': g_peak_struct[4],
            'r_n_peaks': r_peak_struct[0], 'r_dt_main_to_secondary_peak': r_peak_struct[1],
            'r_dmag_secondary_peak': r_peak_struct[2], 'r_secondary_peak_prominence': r_peak_struct[3],
            'r_secondary_peak_width': r_peak_struct[4],
            'g_max_rolling_variance': g_rollvar[0], 'g_mean_rolling_variance': g_rollvar[1],
            'r_max_rolling_variance': r_rollvar[0], 'r_mean_rolling_variance': r_rollvar[1],
            'g_rise_local_curvature': g_rise_curv,
            'g_decline_local_curvature': g_decline_curv,
            'r_rise_local_curvature': r_rise_curv,
            'r_decline_local_curvature': r_decline_curv,
        }

        base_row['features_valid'] = all(not np.isnan(base_row[k]) for k in [
            'g_peak_time', 'g_rise_time', 'g_decline_time', 'g_duration_above_half_flux',
            'r_peak_time', 'r_rise_time', 'r_decline_time', 'r_duration_above_half_flux',
            'mean_g-r', 'g-r_at_g_peak', 'mean_color_rate', 'g_rise_local_curvature', 'g_decline_local_curvature',
            'r_rise_local_curvature', 'r_decline_local_curvature']
        )

        if not return_uncertainty:
            return pd.DataFrame([base_row])
        results = []
        for _ in range(n_trials):
            perturbed_g = self.g['mag'] + np.random.normal(0, self.g['err'])
            perturbed_r = self.r['mag'] + np.random.normal(0, self.r['err'])
            f = SupernovaFeatureExtractor(self.ZTFID, self.g['time'], perturbed_g, self.g['err'], self.r['time'], perturbed_r, self.r['err'])
            results.append(f.extract_features().iloc[0])
        df = pd.DataFrame(results)
        uncertainty = df.std().add_suffix('_err')
        return pd.DataFrame([{**base_row, **uncertainty.to_dict()}])
