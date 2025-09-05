import numpy as np

def filter_pdw_data(pdw_data, filter_cf_zero=True, filter_outliers=True, outlier_percentile=1):
    """
    Applies consistent filtering to PDW data.
    
    Args:
        pdw_data (np.ndarray): The raw PDW data.
        filter_cf_zero (bool): If True, removes pulses with Carrier Frequency of 0.
        filter_outliers (bool): If True, removes outliers based on percentile.
        outlier_percentile (int): The percentile to use for outlier removal (e.g., 1 means 1st and 99th).

    Returns:
        np.ndarray: The filtered PDW data.
    """
    if pdw_data.size == 0:
        return pdw_data

    original_count = len(pdw_data)
    print(f"Original pulse count: {original_count}")
    
    if filter_cf_zero:
        pdw_data = pdw_data[pdw_data[:, 1] != 0]
        print(f"Filtered pulses with CF=0. Remaining: {len(pdw_data)}")

    if filter_outliers and len(pdw_data) > 0:
        # Select features for outlier detection (CF, PW, Amp)
        features_for_outlier_detection = pdw_data[:, [1, 2, 4]]
        q_low = np.percentile(features_for_outlier_detection, outlier_percentile, axis=0)
        q_high = np.percentile(features_for_outlier_detection, 100 - outlier_percentile, axis=0)
        
        mask = np.all((features_for_outlier_detection >= q_low) & (features_for_outlier_detection <= q_high), axis=1)
        pdw_data = pdw_data[mask]
        print(f"Filtered outliers using {outlier_percentile}th percentile. Remaining: {len(pdw_data)}")

    if len(pdw_data) == 0:
        print("WARNING: No pulses remain after filtering.")

    return pdw_data
