# Quite messy memorandum

Quick map of what happens when I run `TDT_main.m` on fresh TDT photometry data.

## Execution order from TDT_main.m
1) `TDT_demod(session_name, Save_univ_dir0, Save_univ_dir1, reload)`  
   - Checks if Stage0 (`Session_<folder>_UnivRAW_stage0.mat`) already exists in `Save_univ_dir0`; if not, calls `TDT_read_stage0` to read the TDT tank (via `TDTbin2mat`) and save Stage0.  
   - Demodulates each fiber/channel (`quickdemod`), extracts epocs/behavior events, and gathers stream metadata.  
   - Saves Stage1 (`Session_<folder>_UnivRAW_offdemod.mat`) into `Save_univ_dir1` and also into a dated subfolder.  
   - Helper functions used: `TDT_read_stage0.m`, `extract_Box_name.m`, `quickdemod` (from pMAT), `lowpass_butter_sos_gain1.m`.

2) `TDT_dFF_stage2(session_name, Stem_Dir, Save_univ_dir0, Save_univ_dir1, Save_univ_dir2, chunky_or_not)`  
   - Loads Stage1 (`Session_<folder>_UnivRAW_offdemod.mat`); if missing, re-runs `TDT_demod` with `reload=1`.  
   - Low-pass filters demodulated signals, computes dF/F for multiple control/signal pairs, and normalizes timestamps.  
   - Writes Stage2 dF/F output: `Session_<folder>_dFF.mat` into `Save_univ_dir2`. Also uses `lowpass_butter_sos_gain1.m` and `DeltaF` (from pMAT toolkits).

## Inputs
- `session_name`: full path to the TDT tank/animal folder (e.g., `.../Tanks/.../Mouse-YYYYMM-DDHHMM`).
- `Save_univ_dir0/1/2`: where Stage0, Stage1, and Stage2 outputs are written.
- `Stem_Dir`: root for additional outputs.
- `reload`: set to 1 to force recomputation of Stage1 even if it exists.
- `chunky_or_not`: passed through to Stage2 (chunked processing flag).

## Outputs
- Stage0: `Session_<folder>_UnivRAW_stage0.mat` (raw TDT readout/cache).
- Stage1: `Session_<folder>_UnivRAW_offdemod.mat` (demodulated channels, events, metadata).
- Stage2: `Session_<folder>_dFF.mat` (dF/F per box/fiber with trial start info).

## Notes and dependencies
- Make sure MATLAB’s path points to this folder and `pMAT v1-2 MATLAB`(I damped literally all the possible path with subfolders in anyway).
- Channel/fiber naming assumes four boxes; warnings are issued if fewer are found (i.e. k00_k00_k00_k00 is correct but if you put like k00_k00_k00 or k00_k00_k00_k00_k00, then you screwed).
