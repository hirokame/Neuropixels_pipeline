# A pipeline for ephys data (from pre-processing to post-processing)

This folder was created by Jiarui Sun.

From pre-processing to post-processing of ephys data, we have a pipeline as below:

* Pre-processing:
  * CatGt: For concatenation of multiple files, delay adjustment, and edge-detection of square wave input for TPrime processing. 
* Spike sorting
  * Kilosort4: Have built-in highpass filter and internal preprocess module
* Post-processing
  * Bombcell: automated curation (parameter adjustment --> Phy manual observation * few times). ***Can classify the cell types within cortex, striatum and GPi(in progress) after update***
  * Phy: manual curation (optional?)
  * Spikeinterface: extract waveforms **(behavioral and tagging period)** and **calculate convolution similarity** to filter out the artifacts (~75-80% non artifact)
  * Unitmatch: match the units across sessions. **RawWaveform extraction: using only the behavioral session spikes (or spikes in the session you want) recommended for lower false negative (better self matching)**



## Pipeline

#### Spikeinterface (for waveform extraction and comparison)

* Input
  * **Sorting:** Kilosort4 output folder (for reading out spike time, end time and stimulation taking off time directly)
  * **Recording:** raw binary data
* Output
  * Waveform visualization
  * non_artifact_units.csv (write in the units with behavior/tag convolution similarity > threshold set)
    * can be used to take the intersection with tagged_units.csv (with high z-scores)

#### Unitmatch

* Input

  * **Sorting:** Kilosort4 output folder

  * **Recording:** raw binary data4

* Output

  * **(the only useful one for following post-processing)** Unitmatch.mat
  * Some other .fig and .mat (not used. And due to the matching of mua, evaluation can't be performed because of certain unit properties needed)

Under ***Unitmatch.mat***, there is an useful struct named ***UniqueIDConversion***. Open Unitmatch.mat directly and you can see UniqueIDConversion in the variable working space, then run ***loadUniqueIDConversion.m*** to get some .csv, and then run ***UMSummary.m*** to get the final 

***UnitMatch_Summary_Sorted.csv,*** with columns  below: (one example row displayed)

| UniqueID | RecSes_ClusID_Pairs                                          | Session_Coverage | Session_List                        | Number_of_Units |
| -------- | ------------------------------------------------------------ | ---------------- | ----------------------------------- | --------------- |
| 1        | [(1, 0), (2, 22), (3, 28), (4, 52), (5, 52), (6, 34), (6, 57), (7, 45),  (8, 46), (9, 85), (10, 86), (11, 60)] | 11               | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] | 12              |

Using ***UnitMatch_Summary_Sorted.csv***, we can plot the psth under a same UID to visualize the matching.

***Note:*** Though Unitmatch has a version in Python, Matlab version is strongly recommended for better high-throughput data processing, serial loading and signal processing.



### File organization (Important for successful code running)

* ***NO*** filefolders or sub-filefolders with the string ***"RawWaveform"*** under the KS4 output ***OTHER THAN*** the one extracted for unitmatch pipeline (should be of the shape (wavelength, channel_num, 2), namely (61, 384, 2))
* put the imec0 files, KS4 and KSqMetrics folders under a single folder (for unknown reasons, .nidq shouldn't be placed under a same folder with imec0 files for spikeinterface read in)


---

## Post-analysis & Behavioral Modeling

After preprocessing, neural and behavioral data are integrated and analyzed using the `postanalysis` module.

### Experimental Paradigm: 4-Port Reversal Learning
The mice perform a spatial reversal learning task in a 4-port arena. 
- **The Rule**: Mice must transition between ports in a specific spatial sequence (e.g., Clockwise) to obtain rewards.
- **The Reversal**: Periodically, the reward rule "reverses" to the opposite direction (e.g., Counter-Clockwise), requiring the mouse to adapt its behavioral strategy.

### Analysis Methodology
Our pipeline implements a high-precision approach to behavioral alignment:
1. **Kinematic-Based Firing Rates**: Instead of averaging activity over whole trials, we segment behavior into **Kinematic States** (clean running trajectories between ports). This isolates neural activity during active movement from consumption or stationary periods at the ports.
2. **Behavioral Switch Alignment**: We distinguish between the *environmental rule switch* (when the computer changes the rule) and the *behavioral switch* (when the mouse actually discovers the new rule). We align analyses to the **Decision Point** (departure from the previous port) and **Success Point** (arrival at the target) of the first correct trial following a reversal.
3. **Strategy Tuning**: We calculate a strategy index (Strategy Tuning) comparing firing rates during CW vs. CCW trajectories to track how neural population representations adapt across switches.

### How to Run
The main entry point for post-analysis is [run_post_analysis.py](file:///e:/Neuropixels/Python/postanalysis/run_post_analysis.py).

```bash
# Run Reversal Dynamics analysis
python postanalysis/run_post_analysis.py --mouse 1818 --day 09182025 --analysis reversal

# Run all behavioral/neural analyses
python postanalysis/run_post_analysis.py --mouse 1818 --day 09182025 --analysis all
```

For detailed documentation on the postanalysis infrastructure, see the [postanalysis README](file:///e:/Neuropixels/Python/postanalysis/README.md).