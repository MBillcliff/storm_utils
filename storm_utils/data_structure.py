"""
Improved ForecastingDataset for time series forecasting with ensemble methods.

Key improvements:
- Better performance with vectorized operations
- Comprehensive documentation
- Type hints for clarity
- Robust error handling
- Configurable constants
- Improved memory efficiency
"""

import bisect
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastingConfig:
    """Configuration constants for the ForecastingDataset."""
    
    # Default storm threshold (Hp30 index)
    DEFAULT_STORM_THRESHOLD = 4.5
    
    # Time conversion (30 min per step)
    MINUTES_PER_STEP = 30
    STEPS_PER_HOUR = 2
    
    # Default window parameters (in hours)
    DEFAULT_INPUT_WINDOW_HOURS = 24
    DEFAULT_BUFFER_DAYS = 180
    
    # OMNI column groups for easy subsetting
    OMNI_COLUMNS = [
        "imf_sc_id", "plasma_sc_id",
        "n_imf", "n_plasma",
        "B_scalar", "B_vector",
        "B_lat", "B_lon",
        "Bx_GSE", "By_GSE", "Bz_GSE",
        "By_GSM", "Bz_GSM",
        "RMS_mag", "RMS_vec",
        "RMS_Bx", "RMS_By", "RMS_Bz",
        "T_sw", "n_sw", "V_sw",
        "flow_lon", "flow_lat",
        "alpha_ratio",
        "sigma_T", "sigma_n", "sigma_V",
        "sigma_phi", "sigma_theta", "sigma_ratio",
        "flow_pressure", "E_field",
        "plasma_beta", "alfven_mach", "magnetosonic_mach",
        "quasi_invariant",
        "Kp", "sunspot_R", "Dst", "ap",
        "f107", "AE", "AL", "AU",
        "pc_index", "lyman_alpha",
        "pflux_gt1", "pflux_gt2", "pflux_gt4",
        "pflux_gt10", "pflux_gt30", "pflux_gt60",
        "flux_flag",
        "Boyle", "Newell", "theta_c", "theta_boyle",
    ]

    SW_TYPE_COLUMNS = ['ICME_flag', 'MC_flag', 'SIR_flag']


class ForecastingDataset(Dataset):
    """
    PyTorch Dataset for ensemble-based solar wind forecasting.
    
    This dataset handles:
    - Multiple ensemble members from HUXt simulations
    - OMNI observational data
    - Storm/non-storm balancing
    - Discontinuity filtering
    - Various train/test split strategies
    
    Args:
        parquet_path: Path to the main data parquet file
        discontinuity_path: Optional path to discontinuity timestamps
        seed: Random seed for reproducibility
        Nens: Number of ensemble members to sample
        lead_time_hours: Forecast lead time in hours
        forecast_duration_hours: Duration of forecast window in hours
        stride_hours: Stride between consecutive windows in hours
        
    Attributes:
        Nens: Number of ensemble members
        valid_indices: List of valid window center indices
        max_targets: Array of maximum target values per window
    """
    
    def __init__(
        self,
        parquet_path: str,
        discontinuity_path: Optional[str] = None,
        seed: int = 42,
        Nens: int = 100,
        lead_time_hours: int = 6,
        forecast_duration_hours: int = 12,
        stride_hours: int = 1
    ):
        logger.info(f"Initializing ForecastingDataset with {Nens} ensembles, "
                   f"{lead_time_hours}h lead time, {forecast_duration_hours}h forecast duration")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        self.seed = seed
        self.Nens = Nens
        
        # Generate ensemble column names
        random_suffixes = np.random.choice(2000, size=Nens, replace=False)
        self.huxt_columns = (
            [f"v_{i}" for i in random_suffixes] +
            [f"vomni_{i}" for i in random_suffixes] +
            [f"vgrad_{i}" for i in random_suffixes]
        )
        
        # Set up OMNI columns
        self.all_omni_columns = ForecastingConfig.OMNI_COLUMNS.copy()
        self.omni_columns = self.all_omni_columns.copy()
        self.target_column = ['hp30']

        # Set up solar wind type columns
        self.sw_type_columns = ForecastingConfig.SW_TYPE_COLUMNS.copy()
        
        # Load data
        self._load_data(parquet_path, discontinuity_path)
        
        # Extract column subsets
        self.v_columns = [col for col in self.df.columns if col.startswith("v_")]
        self.vgrad_columns = [col for col in self.df.columns if col.startswith("vgrad_")]
        self.vomni_columns = [col for col in self.df.columns if col.startswith("vomni_")]
        
        # Convert time parameters to steps
        self.stride = stride_hours * ForecastingConfig.STEPS_PER_HOUR
        self.lead_time = lead_time_hours * ForecastingConfig.STEPS_PER_HOUR
        self.forecast_steps = forecast_duration_hours * ForecastingConfig.STEPS_PER_HOUR
        
        # Define window offsets
        self.input_window_start_offset = -48  # 24 hours back
        self.min_offset = self.input_window_start_offset
        self.max_offset = max(96, self.lead_time + self.forecast_steps)
        
        # Compute valid indices and target statistics
        self.valid_indices = self._compute_valid_indices()
        self.max_targets = self._compute_max_targets()

        self.window_labels = self._get_all_window_labels()
        logger.info(f"Dataset initialized with {len(self.valid_indices)} valid windows")
        
        # SANITY CHECK
        assert len(self.window_labels) == len(self.valid_indices), \
            f"Mismatch: {len(self.window_labels)} labels vs {len(self.valid_indices)} windows"
    
    def _load_data(self, parquet_path: str, discontinuity_path: Optional[str]) -> None:
        """Load data from parquet and discontinuity files.
        
        Creates self.discontinuities and self.discontinuity_indices
        """
        try:
            columns_to_load = self.huxt_columns + self.all_omni_columns + self.sw_type_columns + self.target_column
            logger.info(f"Loading data from {parquet_path}")
            self.df = pd.read_parquet(parquet_path, engine='pyarrow', columns=columns_to_load)
            logger.info(f"Loaded {len(self.df)} time steps")
            
            if discontinuity_path is not None:
                logger.info(f"Loading discontinuities from {discontinuity_path}")
                self.discontinuities = np.load(discontinuity_path, allow_pickle=True)
                self.discontinuity_indices = self.df.index.get_indexer(
                    self.discontinuities, 
                    method='nearest'
                )
                logger.info(f"Loaded {len(self.discontinuities)} discontinuities")
            else:
                self.discontinuities = None
                self.discontinuity_indices = None
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _compute_valid_indices(self) -> Tuple[List[int], List[int]]:
        """
        Compute valid window indices, excluding those crossing discontinuities.
        
        Uses binary search for efficient discontinuity checking.
        
        Returns:
            Tuple of (valid_indices, discontinuity_indices)
        """
        total_points = len(self.df)
        
        # Calculate minimum center index to ensure 27-day recurrence is valid
        # Recurrence window: center + lead_time - 27*48 must be >= 0
        # Therefore: center >= 27*48 - lead_time
        min_for_recurrence = 27 * 48 - self.lead_time
        min_for_input = -self.min_offset
        min_center_idx = max(min_for_recurrence, min_for_input)
        
        # Create candidate indices
        base_indices = list(range(min_center_idx, total_points - self.max_offset, self.stride))
        
        logger.info(f"Min center index: {min_center_idx} (ensures 27-day recurrence >= 0)")
        
        if self.discontinuities is None:
            logger.info("No discontinuities to filter")
            return base_indices, []
        
        logger.info(f"Filtering {len(base_indices)} candidate windows against "
                   f"{len(self.discontinuity_indices)} discontinuities")
        
        # Efficient filtering using binary search
        valid_indices = []
        for center_idx in base_indices:
            start_idx = center_idx + self.min_offset
            end_idx = center_idx + self.max_offset
            
            # Check main window for discontinuities
            left = bisect.bisect_right(self.discontinuity_indices, start_idx)
            right = bisect.bisect_left(self.discontinuity_indices, end_idx)
            
            # Check 27-day recurrence window for discontinuities
            recurrence_start = center_idx + self.lead_time - 27 * 48
            recurrence_end = recurrence_start + self.forecast_steps
            rec_left = bisect.bisect_right(self.discontinuity_indices, recurrence_start)
            rec_right = bisect.bisect_left(self.discontinuity_indices, recurrence_end)
            
            # Valid if NEITHER window contains discontinuities
            if left >= right and rec_left >= rec_right:
                valid_indices.append(center_idx)
        
        logger.info(f"Kept {len(valid_indices)} valid windows after discontinuity filtering")
        return valid_indices
    
    def _compute_max_targets(self) -> np.ndarray:
        """
        Precompute maximum target values for each valid window.
        
        Returns:
            Array of shape (num_windows,) with max target per window
        """
        logger.info("Computing maximum targets for all windows")
        max_targets = []
        
        for idx in self.valid_indices:
            target_start = idx + self.lead_time
            target_end = target_start + self.forecast_steps
            target_slice = self.df.iloc[target_start:target_end][self.target_column].values
            max_targets.append(np.max(target_slice))
        
        return np.array(max_targets)

    def _get_window_label(self, idx: int) -> str:
        """
        Get solar wind classification for a window.
        
        Requires ICME_flag and SIR_flag columns in dataframe.
        
        Parameters
        ----------
        idx : int
            Window position
        
        Returns
        -------
        str
            Classification: 'CME_input', 'CME_forecast', 'SIR_input', 
            'SIR_forecast', 'quiet'
        """
        center_idx = self.valid_indices[idx]
        
        input_start = center_idx + self.min_offset
        input_end = center_idx
        forecast_start = center_idx
        forecast_end = forecast_start + +self.lead_time + self.forecast_steps
        
        # Check flags in each window (fast array slicing)
        icme_flag = self.df.get('ICME_flag', pd.Series(np.zeros(len(self.df)))).values
        sir_flag = self.df.get('SIR_flag', pd.Series(np.zeros(len(self.df)))).values
        mc_flag = self.df.get('MC_flag', pd.Series(np.zeros(len(self.df)))).values
        
        cme_in_input = np.any(icme_flag[input_start:input_end]) or np.any(mc_flag[input_start:input_end])
        cme_in_forecast = np.any(icme_flag[forecast_start:forecast_end]) or np.any(mc_flag[forecast_start:forecast_end])
        sir_in_input = np.any(sir_flag[input_start:input_end])
        sir_in_forecast = np.any(sir_flag[forecast_start:forecast_end])
        
        # Classify with priority
        if cme_in_input:
            return 'ICME_input'
        elif cme_in_forecast:
            return 'ICME_forecast'
        elif sir_in_input:
            return 'SIR_input'
        elif sir_in_forecast:
            return 'SIR_forecast'
        else:
            return 'quiet'
    
    
    def _get_all_window_labels(self,) -> np.ndarray:
        """
        Get labels for all windows in dataset.
        
        Parameters
        ----------
        cache : bool
            If True, cache labels as self._window_labels for faster access
        
        Returns
        -------
        np.ndarray
            Array of labels, length = len(dataset)
        """
        
        # Check if required columns exist
        required_cols = ['ICME_flag', 'SIR_flag']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            logger.warning(f"Missing flag columns: {missing_cols}. "
                          f"All windows will be labeled as 'unknown'. "
                          f"Run preprocessing with add_event_flags=True to add these columns.")
            return np.array(['unknown'] * len(self))
        
        logger.info(f"Labeling {len(self)} windows...")
        
        labels = np.array([self._get_window_label(i) for i in range(len(self))])
        
        # Print statistics
        from collections import Counter
        label_counts = Counter(labels)
        
        logger.info(f"\nWindow Classification Summary:")
        for label, count in label_counts.most_common():
            logger.info(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
        
        return labels
        
    
    def set_omni_columns(self, columns: List[str]) -> None:
        """
        Set which OMNI columns to use as additional features.
        
        Note: V ensemble is always included in MLP features. This controls
        which OMNI parameters to add on top of V.
        
        Args:
            columns: List of OMNI column names to add.
                    Empty list [] means no OMNI features (V ensemble only)
            
        Raises:
            ValueError: If any column is not in the dataset
        """
         
        # Handle no OMNI features
        if columns == [] or columns is None:
            self.omni_columns = []
            logger.info("Set OMNI columns to: [] (V ensemble only, no additional OMNI)")
            return
        
        # Validate columns
        for col in columns:
            if col not in self.all_omni_columns:
                raise ValueError(f"Column '{col}' not found in available OMNI columns. "
                               f"Available: {self.all_omni_columns}")
        
        self.omni_columns = columns
        logger.info(f"Set OMNI columns to: {columns} (V ensemble + these OMNI features)")

    
    def filter_indices_by_storm_strength(
        self,
        subset_indices: List[int],
        min_strength: float,
        max_strength: Optional[float] = None
    ) -> np.ndarray:
        """
        Filter indices by storm strength range.
        
        Args:
            subset_indices: Indices to filter
            min_strength: Minimum storm strength (inclusive)
            max_strength: Maximum storm strength (inclusive), or None for no upper bound
            
        Returns:
            Array of filtered indices
        """
        max_strength = max_strength if max_strength is not None else np.inf
        subset_indices = np.array(subset_indices)
        strengths = self.max_targets[subset_indices]
        
        mask = (strengths >= min_strength) & (strengths <= max_strength)
        filtered = subset_indices[mask]
        
        logger.info(f"Filtered {len(subset_indices)} to {len(filtered)} indices "
                   f"in range [{min_strength}, {max_strength}]")
        return filtered
    
    def filter_indices_by_storm_strength_and_balance(
        self,
        subset_indices: List[int],
        min_strength: float,
        max_strength: Optional[float] = None,
        low_strength: float = ForecastingConfig.DEFAULT_STORM_THRESHOLD
    ) -> np.ndarray:
        """
        Filter by storm strength and balance with random non-storm samples.
        
        Args:
            subset_indices: Indices to filter
            min_strength: Minimum storm strength for "storm" class
            max_strength: Maximum storm strength, or None
            low_strength: Threshold below which samples are "non-storm"
            
        Returns:
            Array of balanced indices (storms + equal number of non-storms)
        """
        max_strength = max_strength if max_strength is not None else np.inf
        subset_indices = np.array(subset_indices)
        strengths = self.max_targets[subset_indices]
        
        # Get storm indices in range
        storm_mask = (strengths >= min_strength) & (strengths <= max_strength)
        storm_indices = subset_indices[storm_mask]
        
        # Get non-storm indices
        non_storm_mask = strengths < low_strength
        non_storm_indices = subset_indices[non_storm_mask]
        
        # Sample equal number of non-storms
        n_samples = min(len(storm_indices), len(non_storm_indices))
        
        np.random.seed(self.seed)
        non_storm_sample = np.random.choice(non_storm_indices, size=n_samples, replace=False)
        
        balanced = np.sort(np.concatenate([storm_indices, non_storm_sample]))
        
        logger.info(f"Balanced dataset: {len(storm_indices)} storms + "
                   f"{len(non_storm_sample)} non-storms = {len(balanced)} total")
        return balanced

    def filter_indices_by_event_type(
        self,
        subset_indices: List[int],
        event_types: Optional[List[str]] = None,
        exclude_quiet: bool = False,
        forecast_only: bool = False,
        input_only: bool = False
    ) -> np.ndarray:
        """
        Filter indices by solar wind event type.
        
        Args:
            subset_indices: Indices to filter
            event_types: List of specific event types to include. Options:
                        'ICME', 'SIR', 'ICME_input', 'ICME_forecast', 
                        'SIR_input', 'SIR_forecast', 'quiet'
                        If None, includes all events
            exclude_quiet: If True, exclude quiet periods
            forecast_only: If True, only include forecast period events
            input_only: If True, only include input period events
            
        Returns:
            Array of filtered indices
            
        Raises:
            ValueError: If event labels are not available
            
        Examples:
            # Get only ICME events (both input and forecast)
            icme_indices = dataset.filter_indices_by_event_type(indices, event_types=['ICME'])
            
            # Get SIR events during forecast period only
            sir_forecast = dataset.filter_indices_by_event_type(
                indices, event_types=['SIR'], forecast_only=True
            )
            
            # Get all non-quiet events
            active = dataset.filter_indices_by_event_type(indices, exclude_quiet=True)
        """
        if self.window_labels is None:
            raise ValueError("Event labels not available. Ensure ICME_flag and SIR_flag "
                            "columns are present in the dataset.")
        
        subset_indices = np.array(subset_indices)
        subset_labels = [self.window_labels[idx] for idx in subset_indices]
        
        # Build mask
        mask = np.ones(len(subset_indices), dtype=bool)
        
        # Filter by specific event types
        if event_types is not None:
            event_mask = np.zeros(len(subset_indices), dtype=bool)
            
            for event_type in event_types:
                if event_type == 'ICME':
                    # Include both ICME_input and ICME_forecast
                    event_mask |= np.array([label in ['ICME_input', 'ICME_forecast'] 
                                           for label in subset_labels])
                elif event_type == 'SIR':
                    # Include both SIR_input and SIR_forecast
                    event_mask |= np.array([label in ['SIR_input', 'SIR_forecast'] 
                                           for label in subset_labels])
                else:
                    # Specific label match
                    event_mask |= np.array([label == event_type for label in subset_labels])
            
            mask &= event_mask
        
        # Filter by period
        if forecast_only:
            mask &= np.array([label.endswith('_forecast') or label == 'quiet' 
                             for label in subset_labels])
        
        if input_only:
            mask &= np.array([label.endswith('_input') or label == 'quiet' 
                             for label in subset_labels])
        
        # Exclude quiet
        if exclude_quiet:
            mask &= np.array([label != 'quiet' for label in subset_labels])
        
        filtered = subset_indices[mask]
        
        logger.info(f"Filtered {len(subset_indices)} to {len(filtered)} indices "
                   f"by event type (types={event_types}, exclude_quiet={exclude_quiet}, "
                   f"forecast_only={forecast_only}, input_only={input_only})")
        
        return filtered
    
    def balance_storms(
        self,
        threshold: float = ForecastingConfig.DEFAULT_STORM_THRESHOLD,
        inplace: bool = True,
        random_state: Optional[int] = None
    ) -> Optional[List[int]]:
        """
        Balance storm and non-storm samples by downsampling the majority class.
        
        Args:
            threshold: Storm threshold
            inplace: If True, update self.valid_indices; if False, return balanced indices
            random_state: Random seed for sampling
            
        Returns:
            Balanced indices if inplace=False, else None
        """
        random_state = random_state if random_state is not None else self.seed
        np.random.seed(random_state)
        
        # Split into storms and non-storms
        storm_mask = self.max_targets >= threshold
        storm_indices = [self.valid_indices[i] for i in np.where(storm_mask)[0]]
        non_storm_indices = [self.valid_indices[i] for i in np.where(~storm_mask)[0]]
        
        # Balance by sampling
        n_samples = min(len(storm_indices), len(non_storm_indices))
        storm_sample = np.random.choice(storm_indices, n_samples, replace=False)
        non_storm_sample = np.random.choice(non_storm_indices, n_samples, replace=False)
        
        balanced_indices = sorted(np.concatenate([storm_sample, non_storm_sample]))
        
        logger.info(f"Balanced dataset: dropped {len(non_storm_indices) - n_samples} non-storm samples")
        logger.info(f"Result: {n_samples} storms + {n_samples} non-storms = {len(balanced_indices)} total")
        
        if inplace:
            # Update valid_indices, max_targets, AND window_labels
            index_map = {idx: i for i, idx in enumerate(self.valid_indices)}
            keep_positions = [index_map[idx] for idx in balanced_indices]  # NEW
            
            self.valid_indices = balanced_indices
            self.max_targets = np.array([self.max_targets[index_map[idx]] for idx in balanced_indices])
            self.window_labels = self.window_labels[keep_positions]
            
            return None
        else:
            return balanced_indices

    def remove_cmes(
        self,
        inplace: bool = True,
    ) -> Optional[List[int]]:
        """
        Remove any windows that contain a CME/ICME event (either in input or forecast window).
    
        Parameters
        ----------
        dataset : ForecastingDataset
            The dataset object to filter.
        inplace : bool
            If True, update dataset.valid_indices, dataset.max_targets, and
            dataset.window_labels in place. If False, return the filtered indices.
    
        Returns
        -------
        None if inplace=True, else list of valid indices with CMEs removed.
        """
        if self.window_labels is None:
            raise ValueError(
                "window_labels not available. Ensure ICME_flag and SIR_flag columns "
                "are present in the dataset."
            )
    
        # Find positions where the label does NOT contain ICME
        cme_labels = {'ICME_input', 'ICME_forecast'}
        keep_positions = [
            i for i, label in enumerate(self.window_labels)
            if label not in cme_labels
        ]
        keep_indices = [self.valid_indices[i] for i in keep_positions]
    
        n_removed = len(self.valid_indices) - len(keep_indices)
        logger.info(
            f"remove_cmes: removed {n_removed} windows containing CME/ICME events. "
            f"{len(keep_indices)} windows remaining."
        )
        logger.info(
            f"Removed {n_removed} CME windows "
            f"({n_removed / len(self.valid_indices) * 100:.1f}%). "
            f"{len(keep_indices)} windows remaining."
        )
    
        if inplace:
            self.valid_indices  = keep_indices
            self.max_targets    = self.max_targets[keep_positions]
            self.window_labels  = self.window_labels[keep_positions]
    
            # Sanity check
            assert len(self.valid_indices) == len(self.max_targets) == len(self.window_labels), \
                "Mismatch after remove_cmes — check index alignment."
            return None
        else:
            return keep_indices
    
    def rotation_aligned_train_test_split(
        self,
        train_ratio: float = 0.8,
        test_fold: int = 0
    ) -> Tuple[List[int], List[int]]:
        """
        Split data into train/test sets aligned with solar rotation periods.
        
        Uses discontinuities to define blocks, then assigns blocks to folds.
        
        Args:
            train_ratio: Proportion of data for training
            test_fold: Which fold to use as test set
            
        Returns:
            Tuple of (train_indices, test_indices) as position indices
        """
        k = int(1 / (1 - train_ratio))
        
        if test_fold >= k:
            logger.warning(f"test_fold {test_fold} >= {k} folds, using test_fold=0")
            test_fold = 0
        
        # Define blocks
        blocks = [0] + self.discontinuity_indices
        
        def block_index(num: int) -> Optional[int]:
            """Find which block a number belongs to."""
            for i in range(len(blocks) - 1):
                if blocks[i] <= num < blocks[i + 1]:
                    return i
            return None
        
        # Map raw indices to positions
        index_to_position = {idx: i for i, idx in enumerate(self.valid_indices)}
        
        train_indices = []
        test_indices = []
        
        for idx in self.valid_indices:
            block_idx = block_index(idx)
            if block_idx is not None:
                pos = index_to_position[idx]
                if block_idx % k == test_fold:
                    test_indices.append(pos)
                else:
                    train_indices.append(pos)
        
        train_pct = len(train_indices) / (len(train_indices) + len(test_indices)) * 100
        logger.info(f"Split: {len(train_indices)} train, {len(test_indices)} test "
                   f"({train_pct:.1f}% train)")
        
        return train_indices, test_indices
    
    def chronological_train_test_split(
        self,
        test_fold: int = 0,
        n_folds: int = 5,
        buffer_days: int = ForecastingConfig.DEFAULT_BUFFER_DAYS
    ) -> Tuple[List[int], List[int]]:
        """
        Chronological train/test split with temporal buffer zones.
        
        Args:
            test_fold: Which temporal fold to use as test
            n_folds: Total number of folds
            buffer_days: Size of buffer zone in days
            
        Returns:
            Tuple of (train_indices, test_indices) as position indices
        """
        valid_indices = np.array(self.valid_indices)
        max_index = valid_indices[-1]
        
        buff_size = buffer_days * 24 * ForecastingConfig.STEPS_PER_HOUR
        chunk_size = (max_index + buff_size) // n_folds
        
        # Define test and buffer intervals
        test_start = chunk_size * test_fold
        test_end = chunk_size * (test_fold + 1) - buff_size
        buff_start = test_start - buff_size
        buff_end = test_end + buff_size
        
        # Filter indices
        test_mask = (valid_indices >= test_start) & (valid_indices <= test_end)
        buff_mask = ((valid_indices >= buff_start) & (valid_indices < test_start)) | \
                    ((valid_indices > test_end) & (valid_indices <= buff_end))
        
        test = valid_indices[test_mask]
        train = valid_indices[~(test_mask | buff_mask)]
        
        # Map to positions
        index_to_position = {idx: i for i, idx in enumerate(self.valid_indices)}
        train_indices = [index_to_position[x] for x in train]
        test_indices = [index_to_position[x] for x in test]
        
        logger.info(f"Chronological split: {len(train_indices)} train, "
                   f"{len(test_indices)} test, buffer={buffer_days} days")
        
        return train_indices, test_indices
    
    def __len__(self) -> int:
        """Return number of valid windows."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single forecast window with all features and targets.
        
        Args:
            idx: Window index (0 to len(dataset)-1)
            
        Returns:
            Dictionary containing:
                - v: Velocity ensemble (full window)
                - v_input: Velocity ensemble (input window only)
                - vgrad: Velocity gradient ensemble
                - vomni: OMNI velocity ensemble
                - omni: OMNI features
                - omni_sw: Solar wind velocity from OMNI
                - historic_target: Historical target values
                - target: Target values in forecast window
                - max_target: Maximum target value in forecast window
                - 27_day_target: 27-day recurrence target
                - center_idx: Center index of window (T0)
                - omni_plotting: OMNI data for full window
                - omni_sw_plotting: Solar wind velocity for full window
                - target_plotting: Target values for full window
        """
        center_idx = self.valid_indices[idx]
        window_label = self.window_labels[idx]
        
        # Define window boundaries
        full_start = center_idx + self.min_offset
        full_end = center_idx + self.max_offset
        input_start = full_start
        input_end = center_idx
        forecast_start = center_idx + self.lead_time
        forecast_end = forecast_start + self.forecast_steps
        recurrence_start = forecast_start - 27 * 48  # 27 days
        recurrence_end = recurrence_start + self.forecast_steps
        
        # Extract data slices
        v_data = self.df.iloc[full_start:full_end][self.v_columns].values
        v_input_data = self.df.iloc[input_start:input_end][self.v_columns].values
        vgrad_data = self.df.iloc[full_start:full_end][self.vgrad_columns].values
        vomni_data = self.df.iloc[input_start:input_end][self.vomni_columns].values
        omni_data = self.df.iloc[input_start:input_end][self.omni_columns].values
        omni_sw_data = self.df.iloc[input_start:input_end]['V_sw'].values
        target_data = self.df.iloc[forecast_start:forecast_end][self.target_column].values
        hist_data = self.df.iloc[input_start:input_end][self.target_column].values
        max_target = self.max_targets[idx]
        recurrence = self.df.iloc[recurrence_start:recurrence_end][self.target_column].values
        
        # Plotting data
        omni_sw_plotting = self.df.iloc[full_start:full_end]['V_sw'].values
        target_plotting = self.df.iloc[full_start:full_end][self.target_column].values
        omni_plotting = self.df.iloc[full_start:full_end][self.omni_columns].values
        
        return {
            'v': v_data.astype(np.float32),
            'v_input': v_input_data.astype(np.float32),
            'vgrad': vgrad_data.astype(np.float32),
            'vomni': vomni_data.astype(np.float32),
            'omni': omni_data.astype(np.float32),
            'omni_sw': omni_sw_data.astype(np.float32),
            'historic_target': hist_data.astype(np.float32),
            'target': target_data.astype(np.float32),
            'max_target': max_target.astype(np.float32),
            '27_day_target': recurrence.astype(np.float32),
            'center_idx': center_idx,
            'omni_plotting': omni_plotting.astype(np.float32),
            'omni_sw_plotting': omni_sw_plotting.astype(np.float32),
            'target_plotting': target_plotting.astype(np.float32),
            'window_label': window_label.astype(str),
        }
    
    def get_storm_statistics(self, threshold: float = ForecastingConfig.DEFAULT_STORM_THRESHOLD) -> Dict:
        """
        Get statistics about storms in the dataset.
        
        Args:
            threshold: Storm threshold
            
        Returns:
            Dictionary with storm statistics
        """
        storm_mask = self.max_targets >= threshold
        n_storms = np.sum(storm_mask)
        n_non_storms = len(self.max_targets) - n_storms
        
        return {
            'n_storms': n_storms,
            'n_non_storms': n_non_storms,
            'storm_percentage': 100 * n_storms / len(self.max_targets),
            'mean_storm_strength': np.mean(self.max_targets[storm_mask]) if n_storms > 0 else 0,
            'max_storm_strength': np.max(self.max_targets) if len(self.max_targets) > 0 else 0,
            'threshold': threshold
        }

    def describe(self, threshold: float = ForecastingConfig.DEFAULT_STORM_THRESHOLD) -> None:
        """
        Print comprehensive statistics about the dataset.
        
        Provides overview of temporal coverage, storm distribution, data quality,
        and feature characteristics.
        
        Parameters
        ----------
        threshold : float
            Storm threshold for classification statistics
        """
        print(f"\n{'='*80}")
        print(f"ForecastingDataset Summary")
        print(f"{'='*80}\n")
        
        # ===== BASIC INFO =====
        print(f"Dataset Configuration:")
        print(f"  Ensemble members (Nens): {self.Nens}")
        print(f"  Lead time: {self.lead_time // 2} hours")
        print(f"  Forecast duration: {self.forecast_steps // 2} hours")
        print(f"  Stride: {self.stride // 2} hours")
        print(f"  Random seed: {self.seed}")
        
        # ===== TEMPORAL COVERAGE =====
        print(f"\nTemporal Coverage:")
        print(f"  Total dataframe length: {len(self.df):,} timesteps")
        print(f"  Dataframe time range: {self.df.index[0]} to {self.df.index[-1]}")
        print(f"  Duration: {(self.df.index[-1] - self.df.index[0]).days} days")
        
        # ===== WINDOW STATISTICS =====
        print(f"\nWindow Statistics:")
        total_possible = (len(self.df) - self.max_offset + self.min_offset) // self.stride
        print(f"  Total possible windows: {total_possible:,}")
        print(f"  Valid windows: {len(self.valid_indices):,}")
        print(f"  Rejected (discontinuities): {total_possible - len(self.valid_indices):,} ({(1 - len(self.valid_indices)/total_possible)*100:.1f}%)")
        print(f"  Discontinuities: {len(self.discontinuity_indices)}")
        
        # Window time coverage
        if len(self.valid_indices) > 0:
            first_window_center = self.df.index[self.valid_indices[0]]
            last_window_center = self.df.index[self.valid_indices[-1]]
            print(f"  Valid window time range: {first_window_center} to {last_window_center}")
            print(f"  Valid window duration: {(last_window_center - first_window_center).days} days")
        
        # ===== STORM STATISTICS =====
        storm_stats = self.get_storm_statistics(threshold=threshold)
        
        print(f"\nStorm Statistics (threshold = {threshold}):")
        print(f"  Storms: {storm_stats['n_storms']:,} ({storm_stats['storm_percentage']:.1f}%)")
        print(f"  Non-storms: {storm_stats['n_non_storms']:,}")
        print(f"  Mean storm strength: {storm_stats['mean_storm_strength']:.2f}")
        print(f"  Max storm strength: {storm_stats['max_storm_strength']:.2f}")
        
        # Distribution of target values
        print(f"\nTarget (Hp30 Max) Distribution:")
        print(f"  Min: {np.min(self.max_targets):.2f}")
        print(f"  25th percentile: {np.percentile(self.max_targets, 25):.2f}")
        print(f"  Median: {np.median(self.max_targets):.2f}")
        print(f"  75th percentile: {np.percentile(self.max_targets, 75):.2f}")
        print(f"  95th percentile: {np.percentile(self.max_targets, 95):.2f}")
        print(f"  Max: {np.max(self.max_targets):.2f}")
        print(f"  Mean: {np.mean(self.max_targets):.2f}")
        print(f"  Std: {np.std(self.max_targets):.2f}")

        print(f"\nSolar Wind Event Classification:")
        
        from collections import Counter
        label_counts = Counter(self.window_labels)
        
        for label, count in label_counts.most_common():
            print(f"  {label}: {count} ({count/len(self.window_labels)*100:.1f}%)")
        
        # ===== FEATURE STATISTICS =====
        print(f"\nFeature Statistics:")
        
        # Sample a window to get feature info
        sample_window = self[0]
        
        print(f"  HUXt features per window:")
        print(f"    v: {sample_window['v'].shape} (timesteps, ensembles)")
        print(f"    vgrad: {sample_window['vgrad'].shape}")
        print(f"    vomni: {sample_window['vomni'].shape}")
        
        print(f"  OMNI features per window:")
        print(f"    Shape: {sample_window['omni'].shape} (timesteps, n_features)")
        print(f"    Selected columns: {len(self.omni_columns)} of {len(self.all_omni_columns)} available")
        if len(self.omni_columns) <= 10:
            print(f"    Current selection: {self.omni_columns}")
        
        print(f"  Target features:")
        print(f"    historic_target: {sample_window['historic_target'].shape}")
        print(f"    target: {sample_window['target'].shape}")
        print(f"    max_target: scalar")
        
        # Total feature count
        v_features = sample_window['v'].size
        vgrad_features = sample_window['vgrad'].size
        vomni_features = sample_window['vomni'].size
        omni_features = sample_window['omni'].size
        hist_features = sample_window['historic_target'].size
        total_features = v_features + vgrad_features + vomni_features + omni_features + hist_features
        
        print(f"\n  Total features per window: {total_features:,}")
        print(f"    v: {v_features:,} ({v_features/total_features*100:.1f}%)")
        print(f"    vgrad: {vgrad_features:,} ({vgrad_features/total_features*100:.1f}%)")
        print(f"    vomni: {vomni_features:,} ({vomni_features/total_features*100:.1f}%)")
        print(f"    omni: {omni_features:,} ({omni_features/total_features*100:.1f}%)")
        print(f"    historic_target: {hist_features:,} ({hist_features/total_features*100:.1f}%)")
        
        # ===== DATA QUALITY =====
        print(f"\nData Quality:")
        
        # Check for NaN in dataframe
        v_cols = [c for c in self.df.columns if c.startswith('v_')]
        if len(v_cols) > 0:
            n_nan_v = self.df[v_cols].isnull().sum().sum()
            print(f"  NaN in velocity columns: {n_nan_v}")
        
        if 'hp30' in self.df.columns:
            n_nan_target = self.df['hp30'].isnull().sum()
            print(f"  NaN in hp30: {n_nan_target}")
        
        omni_cols_in_df = [c for c in self.all_omni_columns if c in self.df.columns]
        if len(omni_cols_in_df) > 0:
            n_nan_omni = self.df[omni_cols_in_df].isnull().sum().sum()
            print(f"  NaN in OMNI columns: {n_nan_omni}")
        
        # ===== MEMORY USAGE =====
        print(f"\nMemory Usage:")
        memory_mb = self.df.memory_usage(deep=True).sum() / (1024**2)
        print(f"  DataFrame: {memory_mb:.1f} MB")
        print(f"  Valid indices: {len(self.valid_indices) * 8 / 1024:.1f} KB")
        print(f"  Max targets: {self.max_targets.nbytes / 1024:.1f} KB")
        print(f"  Total (approx): {memory_mb + (len(self.valid_indices) * 8 + self.max_targets.nbytes) / (1024**2):.1f} MB")
        
        print(f"\n{'='*80}\n")
    
    
    def get_feature_statistics(self, n_samples=100) -> Dict:
        """
        Compute detailed statistics on features by sampling windows.
        
        Parameters
        ----------
        n_samples : int
            Number of random windows to sample for statistics
        
        Returns
        -------
        dict
            Feature statistics including means, stds, and correlations
        """
        # Sample random windows
        n_samples = min(n_samples, len(self))
        sample_indices = np.random.choice(len(self), size=n_samples, replace=False)
        
        # Collect statistics
        v_means = []
        v_stds = []
        omni_v_means = []
        target_means = []
        
        for idx in sample_indices:
            window = self[idx]
            v_means.append(window['v'].mean())
            v_stds.append(window['v'].std())
            omni_v_means.append(window['omni_sw'].mean() if 'omni_sw' in window else np.nan)
            target_means.append(window['target'].mean())
        
        stats = {
            'huxt_velocity': {
                'mean': np.mean(v_means),
                'std': np.std(v_means),
                'min': np.min(v_means),
                'max': np.max(v_means)
            },
            'huxt_variability': {
                'mean': np.mean(v_stds),
                'std': np.std(v_stds),
            },
            'omni_velocity': {
                'mean': np.nanmean(omni_v_means),
                'std': np.nanstd(omni_v_means),
            },
            'target': {
                'mean': np.mean(target_means),
                'std': np.std(target_means),
            }
        }
        
        return stats
    
    
    def plot_dataset_overview(self, threshold: float = ForecastingConfig.DEFAULT_STORM_THRESHOLD):
        """
        Create overview plots of the dataset.
        
        Parameters
        ----------
        threshold : float
            Storm threshold for visualization
        """
        import matplotlib.pyplot as plt
        
        # Determine layout based on whether event labels are available
        if self.window_labels is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
        
        # Panel 1: Distribution of max targets
        ax1 = axes[0]
        max_target = max(self.max_targets)
        bins = np.arange(1/6, max_target, 1/3)
        ax1.hist(self.max_targets, bins=bins, edgecolor='black', alpha=0.7)
        ax1.axvline(threshold, color='red', linestyle='--', lw=2, label=f'Storm threshold ({threshold})')
        ax1.set_xlabel('Max Hp30', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Distribution of Maximum Hp30 per Window', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Panel 2: Temporal distribution of storms
        ax2 = axes[1]
        
        storm_mask = self.max_targets >= threshold
        storm_indices = np.array(self.valid_indices)[storm_mask]
        
        if len(storm_indices) > 0:
            storm_times = [self.df.index[idx] for idx in storm_indices]
            
            # Histogram by year
            storm_years = [t.year for t in storm_times]
            ax2.hist(storm_years, bins=len(set(storm_years)), edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Year', fontsize=11)
            ax2.set_ylabel('Number of Storms', fontsize=11)
            ax2.set_title(f'Storm Distribution Over Time (≥{threshold})', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
        
        # Panel 3: Storm strength vs time
        ax3 = axes[2]
        
        window_times = [self.df.index[idx] for idx in self.valid_indices]
        scatter = ax3.scatter(window_times, self.max_targets, c=self.max_targets, 
                             cmap='Reds', s=5, alpha=0.5, vmin=0, vmax=10)
        ax3.axhline(threshold, color='red', linestyle='--', lw=1.5, alpha=0.7)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Max Hp30', fontsize=11)
        ax3.set_title('Storm Strength Over Time', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Hp30')
        
        # Rotate x-axis labels
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Panel 4: Discontinuity distribution
        ax4 = axes[3]
        
        if len(self.discontinuity_indices) > 0:
            disc_times = [self.df.index[idx] for idx in self.discontinuity_indices]
            disc_years = [t.year for t in disc_times]
            
            ax4.hist(disc_years, bins=len(set(disc_years)), edgecolor='black', alpha=0.7, color='orange')
            ax4.set_xlabel('Year', fontsize=11)
            ax4.set_ylabel('Number of Discontinuities', fontsize=11)
            ax4.set_title('Discontinuities Over Time', fontsize=12, fontweight='bold')
            ax4.grid(alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No discontinuities', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Discontinuities', fontsize=12, fontweight='bold')
        
        # Panel 5: Solar wind event distribution (if available)
        # Get events by year
        ax5 = axes[4]
        event_by_year = {}
        for idx, label in zip(self.valid_indices, self.window_labels):
            year = self.df.index[idx].year
            if year not in event_by_year:
                event_by_year[year] = {
                    'ICME_input': 0, 'ICME_forecast': 0, 
                    'SIR_input': 0, 'SIR_forecast': 0, 
                    'quiet': 0
                }
            event_by_year[year][label] += 1
        
        years = sorted(event_by_year.keys())
        icme_input_counts = [event_by_year[y]['ICME_input'] for y in years]
        icme_forecast_counts = [event_by_year[y]['ICME_forecast'] for y in years]
        sir_input_counts = [event_by_year[y]['SIR_input'] for y in years]
        sir_forecast_counts = [event_by_year[y]['SIR_forecast'] for y in years]
        quiet_counts = [event_by_year[y]['quiet'] for y in years]
        
        # Stacked bar chart
        width = 0.8
        ax5.bar(years, icme_input_counts, width, label='ICME (input)', color='#e74c3c', alpha=0.9)
        ax5.bar(years, icme_forecast_counts, width, 
               bottom=icme_input_counts, 
               label='ICME (forecast)', color='#c0392b', alpha=0.9)
        ax5.bar(years, sir_input_counts, width, 
               bottom=np.array(icme_input_counts) + np.array(icme_forecast_counts),
               label='SIR (input)', color='#3498db', alpha=0.9)
        ax5.bar(years, sir_forecast_counts, width,
               bottom=np.array(icme_input_counts) + np.array(icme_forecast_counts) + np.array(sir_input_counts),
               label='SIR (forecast)', color='#2980b9', alpha=0.9)
        
        ax5.set_xlabel('Year', fontsize=11)
        ax5.set_ylabel('Number of Windows', fontsize=11)
        ax5.set_title('Solar Wind Events Over Time', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # Panel 6: Event type pie chart
        ax6 = axes[5]
        from collections import Counter
        label_counts = Counter(self.window_labels)
        
        colors = {
            'ICME_input': '#e74c3c', 
            'ICME_forecast': '#c0392b',
            'SIR_input': '#3498db', 
            'SIR_forecast': '#2980b9',
            'quiet': '#95a5a6'
        }
        labels_ordered = ['ICME_input', 'ICME_forecast', 'SIR_input', 'SIR_forecast', 'quiet']
        counts = [label_counts.get(label, 0) for label in labels_ordered]
        colors_ordered = [colors[label] for label in labels_ordered]
        
        # Filter out zero counts for cleaner pie chart
        labels_filtered = [l for l, c in zip(labels_ordered, counts) if c > 0]
        counts_filtered = [c for c in counts if c > 0]
        colors_filtered = [colors[l] for l in labels_filtered]
        
        ax6.pie(counts_filtered, labels=labels_filtered, autopct='%1.1f%%', colors=colors_filtered, startangle=90)
        ax6.set_title('Solar Wind Event Distribution', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
        

    def plot_reduced_dataset_overview(self, threshold: float = ForecastingConfig.DEFAULT_STORM_THRESHOLD):
        """
        Create overview plots of the dataset.
        
        Parameters
        ----------
        threshold : float
            Storm threshold for visualization
        """
        import matplotlib.pyplot as plt
        
        # Determine layout based on whether event labels are available
        if self.window_labels is not None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes = axes.flatten()
        
        # Panel 1: Distribution of max targets
        ax1 = axes[0]
        max_target = max(self.max_targets)
        bins = np.arange(1/6, max_target, 1/3)
        ax1.hist(self.max_targets, bins=bins, edgecolor='black', alpha=0.7)
        ax1.axvline(threshold, color='red', linestyle='--', lw=2, label=f'Storm threshold ({threshold})')
        ax1.set_xlabel('Max Hp30', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Distribution of Maximum Hp30 per Window', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Panel 2: Temporal distribution of storms
        ax2 = axes[1]
        
        storm_mask = self.max_targets >= threshold
        storm_indices = np.array(self.valid_indices)[storm_mask]
        
        if len(storm_indices) > 0:
            storm_times = [self.df.index[idx] for idx in storm_indices]
            
            # Histogram by year
            storm_years = [t.year for t in storm_times]
            ax2.hist(storm_years, bins=len(set(storm_years)), edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Year', fontsize=11)
            ax2.set_ylabel('Number of Storms', fontsize=11)
            ax2.set_title(f'Storm Distribution Over Time (≥{threshold})', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
        
        if self.window_labels is not None:
            # Panel 3: Solar wind event distribution over time
            ax3 = axes[2]
            
            event_by_year = {}
            for idx, label in zip(self.valid_indices, self.window_labels):
                year = self.df.index[idx].year
                if year not in event_by_year:
                    event_by_year[year] = {
                        'ICME_input': 0, 'ICME_forecast': 0, 
                        'SIR_input': 0, 'SIR_forecast': 0, 
                        'quiet': 0
                    }
                event_by_year[year][label] += 1
            
            years = sorted(event_by_year.keys())
            icme_input_counts = [event_by_year[y]['ICME_input'] for y in years]
            icme_forecast_counts = [event_by_year[y]['ICME_forecast'] for y in years]
            sir_input_counts = [event_by_year[y]['SIR_input'] for y in years]
            sir_forecast_counts = [event_by_year[y]['SIR_forecast'] for y in years]
            
            # Stacked bar chart
            width = 0.8
            ax3.bar(years, icme_input_counts, width, label='ICME (input)', color='#e74c3c', alpha=0.9)
            ax3.bar(years, icme_forecast_counts, width, 
                   bottom=icme_input_counts, 
                   label='ICME (forecast)', color='#c0392b', alpha=0.9)
            ax3.bar(years, sir_input_counts, width, 
                   bottom=np.array(icme_input_counts) + np.array(icme_forecast_counts),
                   label='SIR (input)', color='#3498db', alpha=0.9)
            ax3.bar(years, sir_forecast_counts, width,
                   bottom=np.array(icme_input_counts) + np.array(icme_forecast_counts) + np.array(sir_input_counts),
                   label='SIR (forecast)', color='#2980b9', alpha=0.9)
            
            ax3.set_xlabel('Year', fontsize=11)
            ax3.set_ylabel('Number of Windows', fontsize=11)
            ax3.set_title('Solar Wind Events Over Time', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            # Panel 4: Event type pie chart
            ax4 = axes[3]
            from collections import Counter
            label_counts = Counter(self.window_labels)
            
            colors = {
                'ICME_input': '#e74c3c', 
                'ICME_forecast': '#c0392b',
                'SIR_input': '#3498db', 
                'SIR_forecast': '#2980b9',
                'quiet': '#95a5a6'
            }
            labels_ordered = ['ICME_input', 'ICME_forecast', 'SIR_input', 'SIR_forecast', 'quiet']
            counts = [label_counts.get(label, 0) for label in labels_ordered]
            colors_ordered = [colors[label] for label in labels_ordered]
            
            # Filter out zero counts for cleaner pie chart
            labels_filtered = [l for l, c in zip(labels_ordered, counts) if c > 0]
            counts_filtered = [c for c in counts if c > 0]
            colors_filtered = [colors[l] for l in labels_filtered]
            
            ax4.pie(counts_filtered, labels=labels_filtered, autopct='%1.1f%%', 
                   colors=colors_filtered, startangle=90)
            ax4.set_title('Solar Wind Event Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    
    def _print_summary(self, threshold: float):
        """Helper method to print text summary."""
        storm_stats = self.get_storm_statistics(threshold)
        
        print(f"\n{'='*80}")
        print(f"Quick Summary")
        print(f"{'='*80}")
        print(f"Windows: {len(self):,} valid out of {len(self.df):,} timesteps")
        print(f"Storms (≥{threshold}): {storm_stats['n_storms']:,} ({storm_stats['storm_percentage']:.1f}%)")
        print(f"Strongest storm: {storm_stats['max_storm_strength']:.2f}")
        print(f"Discontinuities: {len(self.discontinuity_indices)}")
        print(f"{'='*80}\n")
            