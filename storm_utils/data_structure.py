import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from storm_utils.data_loader import load_omni_data


class ForecastingDataset(Dataset):
    def __init__(self, parquet_path, discontinuity_path=None, Nens=100, lead_time_hours=6, forecast_duration_hours=12, stride_hours=1):
        # Generate 'Nens' unique random integers between 0 and 1999
        np.random.seed(42)
        random_suffixes = np.random.choice(2000, size=Nens, replace=False)
        
        self.Nens = Nens
        
        # Format them as string suffixes
        all_columns = ([f"v_{i}" for i in random_suffixes] +
                    [f"vomni_{i}" for i in random_suffixes] +
                    [f"vgrad_{i}" for i in random_suffixes])
        
        # Filter columns that end with one of the chosen suffixes
        columns_to_load = all_columns + ['omni_flow_speed', 'hp30']
        
        # Load only the selected columns
        self.df = pd.read_parquet(parquet_path, engine='pyarrow', columns=columns_to_load)
        if discontinuity_path is not None:
            self.discontinuities = np.load(discontinuity_path, allow_pickle=True)

        self.v_columns = [col for col in self.df.columns if col.startswith("v_")]
        self.vgrad_columns = [col for col in self.df.columns if col.startswith("vgrad_")]
        self.vomni_columns = [col for col in self.df.columns if col.startswith("vomni_")]
        self.omni_column = 'omni_flow_speed'
        self.target_column = 'hp30'

        # Gap between successive windows: 30 min per step
        self.stride = stride_hours * 2 
        
        # Sampling factor: 30 min per step
        self.lead_time = lead_time_hours * 2
        self.forecast_steps = forecast_duration_hours * 2

        #Window Spans
        self.v_window = 72 * 2       # 72h = 144 steps
        self.vgrad_window = 72 * 2   # 72h = 144 step

        # Offsets relative to forecast time t
        self.v_start_offset = -24 * 2
        self.v_end_offset = 48 * 2
        self.vgrad_start_offset = -24 * 2
        self.vgrad_end_offset = 48 * 2
        self.vomni_start_offset = -24 * 2
        self.vomni_end_offset = 0
        self.omni_start_offset = -24 * 2
        self.omni_end_offset = 0
        self.historic_target_start_offset = -24 * 2
        self.historic_target_end_offset = 0
        self.target_start_offset = self.lead_time
        self.target_end_offset = self.lead_time + self.forecast_steps

        self.min_offset = min(self.v_start_offset, self.vgrad_start_offset, self.vomni_start_offset)
        self.max_offset = max(self.v_end_offset, self.vgrad_end_offset, self.target_end_offset)
        self.valid_indices, self.discontinuity_indices = self._compute_valid_indices()
        self.max_targets = self._compute_max_targets()
        

    def _compute_valid_indices(self):
        # Filter out discontinuities
        total_points = len(self.df)
    
        base_indices = range(-self.min_offset, total_points - self.max_offset, self.stride)
    
        if not hasattr(self, 'discontinuities'):
            return list(base_indices), list([])
    
        # Convert Timestamps to integer indices
        ts_to_idx = {ts: i for i, ts in enumerate(self.df.index)}
        discontinuity_indices = sorted(ts_to_idx[ts] for ts in self.discontinuities if ts in ts_to_idx)
        
        valid_indices = []
        for center_idx in base_indices:
            start_idx = center_idx + self.min_offset
            end_idx = center_idx + self.max_offset
            # Reject if any discontinuity lies strictly within the window
            if not any(start_idx < d <= end_idx for d in discontinuity_indices):
                valid_indices.append(center_idx)
        
        return valid_indices, discontinuity_indices
        
    def _compute_max_targets(self):
        max_targets = []
        
        for idx in self.valid_indices:
            target_start = idx + self.target_start_offset
            target_end = idx + self.target_end_offset

            # Slice the dataframe for this window
            target_slice = self.df.iloc[target_start:target_end]
            
            # Values and times
            values = target_slice[self.target_column].values
            
            # Find maximum
            max_idx = np.argmax(values)
            max_targets.append(values[max_idx])
        
        return np.array(max_targets)


    def filter_indices_by_storms_only(self, subset_indices, min_strength, max_strength=None):
        '''given a subset of indices, returns indices where the storm strength lies between specified bounds
            note: will return all indices within the specified bounds'''
        
        max_strength = max_strength if max_strength is not None else np.inf
        return [idx for idx in subset_indices if min_strength <= self.max_targets[idx] <= max_strength]

    def filter_indices_by_storm_strength(self, subset_indices, min_strength, max_strength=None, low_strength=4.66,):
        '''given a set of indices for both storm and non-storm windows, returns indices where storm strength lies
            between specified bounds. Also includes a random subset of non-storms equal in size to number of storms'''
        
        max_strength = max_strength if max_strength is not None else np.inf
        subset_indices = np.array(subset_indices)
        labels = np.arange(len(subset_indices))
        strengths = np.array(self.max_targets)

        
        # Filter indices in the desired range
        in_range_mask = (strengths[subset_indices] >= min_strength) & (strengths[subset_indices] <= max_strength)
        in_range_indices = subset_indices[in_range_mask]
    
        # Filter low-strength indices
        low_strength_mask = strengths[subset_indices] < low_strength
        low_strength_indices = subset_indices[low_strength_mask]
    
        # Combine and return
        return sorted(np.concatenate([in_range_indices, low_strength_indices]))


    def filter_indices_by_storm_strength_and_balance(self, subset_indices, min_strength, max_strength=None, low_strength=4.66,):
        '''given a set of indices for both storm and non-storm windows, returns indices where storm strength lies
            between specified bounds. Also includes a random subset of non-storms equal in size to number of storms'''
        
        max_strength = max_strength if max_strength is not None else np.inf
        subset_indices = np.array(subset_indices)
        labels = np.arange(len(subset_indices))
        strengths = np.array(self.max_targets)

        
        # Filter indices in the desired range
        in_range_mask = (strengths[subset_indices] >= min_strength) & (strengths[subset_indices] <= max_strength)
        in_range_indices = subset_indices[in_range_mask]
    
        # Filter low-strength indices
        low_strength_mask = strengths[subset_indices] < low_strength
        low_strength_indices = subset_indices[low_strength_mask]
        low_strength_indices_subset = np.random.choice(
            low_strength_indices, 
            size=min(len(low_strength_indices), len(in_range_indices)), 
            replace=False
        )
    
        # Combine and return
        return sorted(np.concatenate([in_range_indices, low_strength_indices_subset]))

    def sort_indices_by_storm_strength(self, subset_indices, return_order=False):
        subset_indices = np.array(subset_indices)
        strengths = np.array(self.max_targets)
        
        order = np.argsort([strengths[i] for i in subset_indices])[::-1]
        subset_indices = subset_indices[order]
        
        return subset_indices, order
        

    def rotation_aligned_train_test_split(self, train_ratio=0.8, storm_test_thresh=4.66, test_fold=0):
        k = 1 // (1 - train_ratio)

        if test_fold >= k:
            print('Specified test fold > n_folds. Resorting to test_fold = 0')
            test_fold = 0
        # Specify blocks for splitting
        blocks = [0] + self.discontinuity_indices

        def block_index(num):
            # returns index of block 
            for i in range(len(blocks) - 1):
                if blocks[i] <= num < blocks[i + 1]:
                    return i
            return None

        index_to_position = {idx: i for i, idx in enumerate(self.valid_indices)}

        train_indices = [index_to_position[idx] for idx in self.valid_indices if (i := block_index(idx)) is not None and i % k != test_fold]
        test_indices  = [index_to_position[idx] for idx in self.valid_indices if (i := block_index(idx)) is not None and i % k == test_fold]

        print(f'Train size: {len(train_indices)}, Test size: {len(test_indices)}')
        print(f'Split and filtered into {len(train_indices)/(len(train_indices) + len(test_indices)) * 100:.1f}% Train')
        
        return train_indices, test_indices

    def random_rotation_aligned_train_test_split(self, train_ratio=0.8, storm_test_thresh=4.66, seed=None):
        if seed is not None:
            random.seed(seed)  # optional reproducibility
    
        k = 5  # always split into 5-block groups
    
        # Define blocks based on discontinuities
        blocks = [0] + self.discontinuity_indices
    
        def block_index(num):
            # returns index of block 
            for i in range(len(blocks) - 1):
                if blocks[i] <= num < blocks[i + 1]:
                    return i
            return None
    
        index_to_position = {idx: i for i, idx in enumerate(self.valid_indices)}
    
        train_indices = []
        test_indices = []
    
        # Determine which blocks are test within each 5-block group
        n_blocks = len(blocks) - 1
        for group_start in range(0, n_blocks, k):
            group_blocks = list(range(group_start, min(group_start + k, n_blocks)))
            if not group_blocks:
                continue
            test_fold = np.random.choice(group_blocks)
            for idx in self.valid_indices:
                i = block_index(idx)
                if i is None:
                    continue
                if i in group_blocks:
                    if i == test_fold:
                        test_indices.append(index_to_position[idx])
                    else:
                        train_indices.append(index_to_position[idx])
    
        return train_indices, test_indices



    def balance_storms(self, threshold=5.0, inplace=True, random_state=42):
        """
        Balances the dataset so that the number of storms (>= threshold) and non-storms (< threshold) are equal.

        Args:
            threshold (float): Threshold on max_target to define a storm.
            inplace (bool): Whether to update self.valid_indices and self.max_targets. If False, returns balanced indices.
            random_state (int): Seed for reproducibility.

        Returns:
            If inplace is False, returns the balanced list of indices.
        """
        np.random.seed(random_state)

        # Split indices into storms and non-storms
        storm_indices = [idx for i, idx in enumerate(self.valid_indices) if self.max_targets[i] >= threshold]
        nonstorm_indices = [idx for i, idx in enumerate(self.valid_indices) if self.max_targets[i] < threshold]

        # Determine how many to sample
        n_samples = min(len(storm_indices), len(nonstorm_indices))

        # Sample each class equally
        storm_sample = np.random.choice(storm_indices, n_samples, replace=False)
        nonstorm_sample = np.random.choice(nonstorm_indices, n_samples, replace=False)

        balanced_indices = sorted(np.concatenate([storm_sample, nonstorm_sample]))

        print(f'dropped {len(nonstorm_indices) - n_samples} non-storm times')
        print('There are now', len(storm_sample), 'storms and', len(nonstorm_sample), 'non-storms')

        if inplace:
            # Update valid_indices and associated max_targets
            index_map = {idx: i for i, idx in enumerate(self.valid_indices)}
            self.valid_indices = list(balanced_indices)
            self.max_targets = np.array([self.max_targets[index_map[idx]] for idx in balanced_indices])
        else:
            return list(balanced_indices)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx, return_time=False):
        center_idx = self.valid_indices[idx]
        min_idx = center_idx + self.min_offset
        max_idx = center_idx + self.max_offset
        v_start = center_idx + self.v_start_offset
        v_end = center_idx + self.v_end_offset
        vgrad_start = center_idx + self.vgrad_start_offset
        vgrad_end = center_idx + self.vgrad_end_offset
        vomni_start = center_idx + self.vomni_start_offset
        vomni_end = center_idx + self.vomni_end_offset
        omni_start = center_idx + self.omni_start_offset
        omni_end = center_idx + self.omni_end_offset
        historic_target_start = center_idx + self.historic_target_start_offset
        historic_target_end = center_idx + self.historic_target_end_offset
        target_start = center_idx + self.target_start_offset
        target_end = center_idx + self.target_end_offset

        twentyseven_recurrence_start = target_start - (27 * 48)
        twentyseven_recurrence_end = target_end - (27 * 48)

        v_data = self.df.iloc[v_start:v_end][self.v_columns].values                                                # [144, Nens]
        vgrad_data = self.df.iloc[vgrad_start:vgrad_end][self.vgrad_columns].values                                # [144, Nens]
        vomni_data = self.df.iloc[vomni_start:vomni_end][self.vomni_columns].values                                # [48, Nens]
        omni_data = self.df.iloc[omni_start:omni_end][self.omni_column].values                                     # [48]
        target_data = self.df.iloc[target_start:target_end][self.target_column].values                             # [F]
        historic_target_data = self.df.iloc[historic_target_start:historic_target_end][self.target_column].values  # [48]
        historic_target_data = np.expand_dims(historic_target_data, axis=-1)                                       # [48, 1]
        max_target_data = self.max_targets[idx] # [1]
        
        target_plotting_data = self.df.iloc[min_idx:max_idx][self.target_column].values                           # [144]
        omni_plotting_data = self.df.iloc[min_idx:max_idx][self.omni_column].values                                # [144]
        twentyseven_target_data = self.df.iloc[twentyseven_recurrence_start:twentyseven_recurrence_end][self.target_column].values  # [F]
        # Output some times
        T0 = self.df.index[center_idx]                                                                            # [1]
        F_start = self.df.index[target_start]                                                                      # [1]
        F_end = self.df.index[target_end]                                                                          # [1]

        out = {
            'v': v_data.astype(np.float32),
            'vgrad': vgrad_data.astype(np.float32),
            'vomni': vomni_data.astype(np.float32),
            'omni': omni_data.astype(np.float32),
            'historic_target': historic_target_data.astype(np.float32),
            'target': target_data.astype(np.float32),
            'max_target': max_target_data.astype(np.float32),
            '27_day_target': twentyseven_target_data.astype(np.float32),
            'target_plotting': target_plotting_data.astype(np.float32),
            'omni_plotting': omni_plotting_data.astype(np.float32),
        }
        
        if return_time:
            out['T0'] = T0
            out['target_start'] = F_start
            out['target_end'] = F_end

        return out
        