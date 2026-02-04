---
trigger: always_on
---

1. Data Loading & Schema Protocols
- All file definitions (headers, dtypes, specific column names) are stored in dataset_config.json in the project root.
- Before writing pd.read_csv, np.load, or h5py.File, you must conceptually check dataset_config.json for the file's schema.
- Be careful that the path in dataset_config.json contains absolute full paths, but mouse name, session name etc could be changed depending on which dataset you are analyzing.
- When biulding/refactoring new analysis functions, always look up the dataloader function in ./postanalysis/data_loader_refactored.py, ./postanalysis/data_loader.py, and helper function in the beginning of ./postanalysis/analyses.py. Try to use those pre-made function first, and try not to build new helper.

2. Time Alignment & Synchronization (The "Common Index" Rule)
- Event CSVs are often truncated. Due to data writing lag, the Event CSV files often skip the first ~300 frames. 
- DLC & strobe_seconds.npy: Complete. Row 0 corresponds to Frame 0.
- "Timestamp" column of the event_corner file is not the time. You always have to combine "Index" column for the number of frames, then use strobe_seconds.npy in kilosort4 output folder to reference the absolute second each frame was taken.
- When merging data, usually reindex the Event dataframe to match the full DLC length, filling missing initial rows with NaNs.