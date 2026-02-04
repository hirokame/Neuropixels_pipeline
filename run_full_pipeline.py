#!/usr/bin/env python3
"""
End-to-end driver for the Neuropixels pipeline described in README.md.

What it does (per session):
1) Optionally run CatGT (user template).
2) Optionally run SGLXMetaToCoords.m to make the Kilosort channel map.
3) Optionally launch Kilosort (user template or spikeinterface fallback).
4) Extract digital I/O (NIDQ) to stim/behavioral npys + spike_seconds.
5) Optionally run TPrime to align spike_seconds -> spike_seconds_adj.
6) Optionally build spike_mask (±tol around stim edges).
7) Optionally run BombCell quality metrics (MATLAB).
8) Optionally trigger spikeinterface waveform extraction (user template).
9) Export template_metrics/ACC/ISI and waveform npys from existing analyzers.
10) Run rule-based cell classification and save plots/CSV.

The script is intentionally modular so you can skip steps you run manually
and re-run only the pieces you want.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
SPIKE_TOOLS_ROOT = REPO_ROOT / "pipeline" / "spikeinterface_waveform_extraction"
DEMO_SGLX_ROOT = REPO_ROOT / "DemoReadSGLXData"
for _p in (SPIKE_TOOLS_ROOT, DEMO_SGLX_ROOT):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))


def log(msg: str) -> None:
    print(f"[pipeline] {msg}")


def run_cmd(
    cmd: Union[str, List[str]],
    dry_run: bool = False,
    cwd: Optional[Path] = None,
    use_shell: bool = False,
) -> None:
    display = cmd if isinstance(cmd, str) else " ".join(cmd)
    log("RUN: " + display + (f" (cwd={cwd})" if cwd else ""))
    if dry_run:
        return
    subprocess.run(
        cmd,
        check=True,
        cwd=str(cwd) if cwd else None,
        shell=use_shell,
    )


def normalize_path(p: Path) -> str:
    return str(p).replace("\\", "/")


def discover_sessions(
    data_roots: Iterable[Path],
    session_names: Optional[Iterable[str]],
    kilosort_output_subdir: str,
) -> List[Dict[str, Path]]:
    metas: List[Path] = []
    
    for root in data_roots:
        root = Path(root)
        if session_names:
            for name in session_names:
                session_dir = root / name
                if not session_dir.exists():
                    # Just log debug/info, not error, as it might be in another root
                    continue
                metas.extend(session_dir.rglob("*_imec0/*.ap.meta"))
        else:
            metas.extend(list(root.rglob("*_imec0/*.ap.meta")))

    sessions: List[Dict[str, Path]] = []
    seen: set[Path] = set()
    for meta in metas:
        imec_dir = meta.parent
        session_dir = imec_dir.parent
        if session_dir in seen:
            continue
        seen.add(session_dir)

        # Prefer CatGT tcat output if present, otherwise fall back to t0 or any ap bin.
        ap_bin = (
            next(imec_dir.glob("*tcat.imec0.ap.bin"), None)
            or next(imec_dir.glob("*t0.imec0.ap.bin"), None)
            or next(imec_dir.glob("*_imec0.ap.bin"), None)
        )
        ks_dir_base = imec_dir / "kilosort4"
        ks_dir = ks_dir_base / kilosort_output_subdir if kilosort_output_subdir else ks_dir_base
        session = {
            "session_name": session_dir.name,
            "session_dir": session_dir,
            "imec_dir": imec_dir,
            "ap_meta": meta,
            "ap_bin": ap_bin,
            "ks_dir": ks_dir,
            "ks_qmetrics_dir": ks_dir / "qMetrics",
            "analyzer_beh": imec_dir / "analyzer_beh",
            "analyzer_tag": imec_dir / "analyzer_tag",
        }
        sessions.append(session)

    if not sessions:
        log(f"No sessions found under {data_roots}")
    else:
        log(f"Found {len(sessions)} session(s)")
    return sessions


def run_channel_map(session: Dict[str, Path], matlab_bin: str, dry_run: bool) -> None:
    meta = session["ap_meta"]
    if not meta.exists():
        log(f"Skip channel map for {session['session_name']}: missing meta file")
        return

    chanmap = meta.with_name(f"{meta.stem}_kilosortChanMap.mat")
    if chanmap.exists():
        log(f"[{session['session_name']}] channel map already exists ({chanmap.name}), skipping")
        return

    meta_str = normalize_path(meta)
    workdir = normalize_path(meta.parent)
    script = (
        f"cd('{workdir}');"
        f"addpath('{normalize_path(REPO_ROOT / 'Preprocessing')}');"
        f"SGLXMetaToCoords('{meta_str}',{{1}});exit"
    )
    run_cmd([matlab_bin, "-batch", script], dry_run=dry_run)


def run_kilosort(
    session: Dict[str, Path],
    template: Optional[str],
    python_bin: str,
    dry_run: bool,
    use_spikeinterface: bool,
    ks_highpass_cutoff: float,
    ks_th_universal: float,
    ks_th_learned: float,
) -> None:
    ks_dir = session["ks_dir"]
    if ks_dir.exists() and any(ks_dir.iterdir()):
        log(f"[{session['session_name']}] kilosort4 folder already present, skipping")
        return
    if not template and not use_spikeinterface:
        log(f"[{session['session_name']}] no kilosort command template provided, skipping")
        return

    # Determine run directory for SpikeInterface to avoid double nesting of 'sorter_output'
    # If ks_dir ends with 'sorter_output', pass the parent, because SI/KS4 wrapper appends 'sorter_output'.
    if ks_dir.name == "sorter_output":
        ks_run_dir = ks_dir.parent
    else:
        ks_run_dir = ks_dir

    fmt_args = {
        "python": python_bin,
        "session_name": session["session_name"],
        "session_dir": normalize_path(session["session_dir"]),
        "imec_dir": normalize_path(session["imec_dir"]),
        "ap_bin": normalize_path(session["ap_bin"]) if session["ap_bin"] else "",
        "ap_meta": normalize_path(session["ap_meta"]),
        "ks_dir": normalize_path(ks_dir),
        "ks_run_dir": normalize_path(ks_run_dir),
    }
    if template:
        cmd_str = template.format(**fmt_args)
        run_cmd(shlex.split(cmd_str), dry_run=dry_run)
        return

    # spikeinterface fallback
    ap_path = fmt_args["ap_bin"] or fmt_args["imec_dir"]
    log(f"[{session['session_name']}] Kilosort input: {ap_path}")
    stream_name = "imec0.ap"
    si_script = f"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '8'
import spikeinterface.full as si
from pathlib import Path

ap_bin = r\"{fmt_args['ap_bin']}\"
if ap_bin:
    from neo.rawio.spikeglxrawio import read_meta_file, extract_stream_info
    import probeinterface
    meta_file = Path(ap_bin).with_suffix(".meta")
    meta = read_meta_file(meta_file)
    info = extract_stream_info(meta_file, meta)
    rec = si.BinaryRecordingExtractor(
        file_paths=[ap_bin],
        sampling_frequency=info["sampling_rate"],
        dtype="int16",
        num_channels=info["num_chan"],
        gain_to_uV=info["channel_gains"],
        offset_to_uV=info["channel_offsets"],
    )
    probe = probeinterface.read_spikeglx(meta_file)
    rec = rec.set_probe(probe)
else:
    rec = si.read_spikeglx(r\"{fmt_args['imec_dir']}\", stream_name='{stream_name}')
    nseg = rec.get_num_segments()
    if nseg > 1:
        lens = [rec.get_num_samples(i) for i in range(nseg)]
        max_i = max(range(nseg), key=lambda i: lens[i])
        rec = rec.select_segments([max_i])

si.run_sorter(
    'kilosort4',
    rec,
    folder=r\"{fmt_args['ks_run_dir']}\",
    remove_existing_folder=True,
    verbose=True,
    use_binary_file=False,
    highpass_cutoff={ks_highpass_cutoff},
    Th_universal={ks_th_universal},
    Th_learned={ks_th_learned},
)
""".strip()
    si_script_path = session["imec_dir"] / f"run_ks_{session['session_name']}.py"
    try:
        with open(si_script_path, "w", encoding="utf-8") as f:
            f.write(si_script)
        log(f"[{session['session_name']}] Wrote Kilosort script to {si_script_path}")
    except Exception as e:
        log(f"[{session['session_name']}] Failed to write Kilosort script: {e}")
        return

    si_cmd = [python_bin, str(si_script_path)]
    run_cmd(si_cmd, dry_run=dry_run)


CATGT_DEFAULT_DIR = Path(r"C:\Users\kouhi\CatGT-win")


def run_catgt(
    session: Dict[str, Path],
    template: Optional[str],
    catgt_dir: Optional[Path],
    dry_run: bool,
    catgt_gfix: Optional[str] = None,
    catgt_t_inline: bool = False,
) -> None:
    # Only run CatGT on g0 folders; skip g1 (tagging-only) folders.
    if not session["session_name"].endswith("_g0"):
        log(f"[{session['session_name']}] not a g0 folder, skipping CatGT")
        return

    if session["ap_bin"] and "tcat" in session["ap_bin"].name:
        log(f"[{session['session_name']}] CatGT output already detected ({session['ap_bin'].name}), skipping CatGT")
        return

    # detect if a matching g1 folder exists to decide g flag
    base_name = session["session_name"][:-3]  # drop trailing "_g0"
    sibling_g1 = session["session_dir"].parent / f"{base_name}_g1"
    g_flag = "0,1" if sibling_g1.exists() else "0"

    # If user did not pass a template, build a simple default:
    # runit.bat -dir=<data_root> -run=<session without _gX> -g=<g_flag> -t=0 -prb_fld -prb=0 -ni -ap
    if not template:
        run_base = base_name
        template = (
            "runit.bat -dir={data_root} -run={run_base} -g={g_flag} -t=0 -prb_fld -prb=0 -ni -ap"
        )
        if catgt_gfix:
            template += f" -gfix={catgt_gfix}"
        if catgt_t_inline:
            template += " -t_inline"

    fmt_args = {
        "session_name": session["session_name"],
        "session_dir": normalize_path(session["session_dir"]),
        "data_root": normalize_path(session["session_dir"].parent),
        "imec_dir": normalize_path(session["imec_dir"]),
        "g_flag": g_flag,
        "run_base": run_base,
    }
    cmd_str = template.format(**fmt_args)
    # Prefer absolute path to runit.bat if present
    runit_path = (catgt_dir or CATGT_DEFAULT_DIR) / "runit.bat"
    if runit_path.exists():
        cmd_str = cmd_str.replace("runit.bat", str(runit_path))
    run_cmd(cmd_str, dry_run=dry_run, cwd=catgt_dir or CATGT_DEFAULT_DIR, use_shell=True)


def run_bombcell(
    session: Dict[str, Path],
    matlab_bin: str,
    bombcell_root: Optional[Path],
    dry_run: bool,
) -> None:
    ks_dir = session["ks_dir"]
    imec_dir = session["imec_dir"]
    
    # User wants output in imec_dir / kilosort4qMetrics
    bc_output_dir = imec_dir / "kilosort4qMetrics"
    qlabel_path = bc_output_dir / "templates._bc_unit_labels.tsv"

    if bc_output_dir.exists():
        log(f"[{session['session_name']}] BombCell output already exists, skipping BombCell")
        return
    
    should_run_matlab = not qlabel_path.exists()

    # Auto-detect BombCell path if default is provided and invalid
    bc_path_final = bombcell_root
    if bombcell_root == REPO_ROOT / "Preprocessing":
        log(f"[{session['session_name']}] Checking for BombCell in default path and parent directory...")
        alt_bc_path = REPO_ROOT.parent / "bombcell"
        if alt_bc_path.exists() and (alt_bc_path / "+bc").exists():
            log(f"[{session['session_name']}] Found BombCell at {alt_bc_path}, using it.")
            bc_path_final = alt_bc_path
        else:
             log(f"[{session['session_name']}] Did not find BombCell in parent directory.")

    if should_run_matlab:
        if bc_path_final is None:
            log(f"[{session['session_name']}] no BombCell path provided, skipping")
            return
        if not bc_path_final.exists():
            log(f"[{session['session_name']}] BombCell path not found at {bc_path_final}, skipping")
            return
        if not session["ap_bin"]:
            log(f"[{session['session_name']}] missing ap.bin, skipping BombCell")
            return

        bc_path = normalize_path(bc_path_final)
        ephys_kilo = normalize_path(session["ks_dir"])
        raw_bin = normalize_path(session["ap_bin"])
        bc_out_str = normalize_path(bc_output_dir)

        # The user might have a wrapper in Preprocessing, so add that to the path too.
        preprocessing_path = normalize_path(REPO_ROOT / "Preprocessing")

        script = (
            f"addpath('{preprocessing_path}');"
            f"addpath(genpath('{bc_path}'));"
            f"BombCell('{ephys_kilo}', '{raw_bin}', '{bc_out_str}');"
            "exit"
        )
        run_cmd([matlab_bin, "-batch", script], dry_run=dry_run)
    else:
        log(f"[{session['session_name']}] BombCell outputs present, skipping MATLAB run")

    # Move BombCell outputs (flatten qMetrics if created)
    if not dry_run:
        # If BombCell created a qMetrics subdirectory, move its contents up
        nested_qmetrics = bc_output_dir / "qMetrics"
        if nested_qmetrics.exists():
            log(f"[{session['session_name']}] Flattening nested qMetrics in {bc_output_dir}")
            for item in nested_qmetrics.iterdir():
                dst = bc_output_dir / item.name
                if dst.exists():
                    if dst.is_dir(): shutil.rmtree(dst)
                    else: dst.unlink()
                shutil.move(str(item), str(dst))
            nested_qmetrics.rmdir()

    # No need to move specific files anymore as BombCell saves them to bc_output_dir now



def run_spikeinterface_extract(
    session: Dict[str, Path],
    template: Optional[str],
    python_bin: str,
    dry_run: bool,
    use_spikeinterface: bool,
) -> None:
    fmt_args = {
        "python": python_bin,
        "session_name": session["session_name"],
        "session_dir": normalize_path(session["session_dir"]),
        "imec_dir": normalize_path(session["imec_dir"]),
        "ap_bin": normalize_path(session["ap_bin"]) if session["ap_bin"] else "",
        "ap_meta": normalize_path(session["ap_meta"]),
        "ks_dir": normalize_path(session["ks_dir"]),
    }

    output_dir = session["imec_dir"] / "analyzer_beh"
    if output_dir.exists():
        log(f"[{session['session_name']}] output folder already exists, skipping")
        return
    
    if template:
        cmd_str = template.format(**fmt_args)
        run_cmd(shlex.split(cmd_str), dry_run=dry_run)
        return

    if not use_spikeinterface:
        log(f"[{session['session_name']}] no spikeinterface command template provided, skipping")
        return

    try:
        import spikeinterface_v2.full as si  # type: ignore
    except Exception as exc:
        log(f"[{session['session_name']}] spikeinterface_v2 not available ({exc}), skipping")
        return

    if not session["ap_bin"]:
        log(f"[{session['session_name']}] missing ap.bin; cannot build analyzer")
        return

    # Load main recording using read_spikeglx, which correctly handles metadata
    # like inter_sample_shift required for phase correction.
    try:
        rec_main = si.read_spikeglx(str(session["imec_dir"]), stream_name="imec0.ap")
    except Exception as e:
        log(f"[{session['session_name']}] Error loading SpikeGLX recording: {e}")
        return

    if rec_main.get_num_segments() > 1:
        lens = [rec_main.get_num_samples(i) for i in range(rec_main.get_num_segments())]
        max_i = max(range(len(lens)), key=lambda i: lens[i])
        rec_main = rec_main.select_segments([max_i])

    log(f"[{session['session_name']}] Applying preprocessing...")
    rec_main = si.highpass_filter(rec_main, freq_min=400.)
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec_main)
    if len(bad_channel_ids) > 0:
        rec_main = rec_main.remove_channels(bad_channel_ids)
    if "inter_sample_shift" in rec_main.get_property_keys():
        rec_main = si.phase_shift(rec_main)
    else:
        log(f"[{session['session_name']}] WARNING: 'inter_sample_shift' missing. Skipping phase_shift.")
    rec_main = si.common_reference(rec_main, operator="median", reference="global")
    
    # Workaround for spikeinterface_v2:
    # It looks for qMetrics in ks_dir / (ks_dir.name + "qMetrics") based on the error log.
    # We have the data in imec_dir / kilosort4qMetrics.
    sibling_qmetrics = session["imec_dir"] / "kilosort4qMetrics"
    
    # Construct the path where SI failed: .../kilosort4/kilosort4qMetrics
    buggy_qmetrics_path = session["imec_dir"] / "kilosort4" / "kilosort4qMetrics"
    created_temp_qmetrics = False

    if not buggy_qmetrics_path.exists():
        if sibling_qmetrics.exists():
            log(f"[{session['session_name']}] Workaround: Copying qMetrics from {sibling_qmetrics} to {buggy_qmetrics_path} for SI v2")
            if not dry_run:
                shutil.copytree(sibling_qmetrics, buggy_qmetrics_path)
                created_temp_qmetrics = True
        else:
            log(f"[{session['session_name']}] WARNING: qMetrics not found. SpikeInterface extraction might fail. Run 'bombcell' step first.")

    try:
        # Suppress noisy stdout from spikeinterface_v2 (pandas dtypes etc)
        with open(os.devnull, "w") as fnull:
            _orig_stdout = sys.stdout
            sys.stdout = fnull
            try:
                sorting = si.read_kilosort(str(session["ks_dir"]))
            finally:
                sys.stdout = _orig_stdout

        # 1) analyzer_beh
        if not session["analyzer_beh"].exists():
            an = si.create_sorting_analyzer(
                sorting,
                rec_main,
                format="binary_folder",
                folder=str(session["analyzer_beh"]),
                overwrite=True,
                sparse=True,
                num_spikes_for_sparsity=100,
                method="radius",
                radius_um=100.0,
                peak_sign="neg",
                n_jobs=12,
                chunk_duration="5s",
                progress_bar=True,
            )
            an.compute("random_spikes")
            an.compute("waveforms", n_jobs=12, chunk_duration="5s", progress_bar=True)
            an.compute("templates", operators=["average", "median"])
            log(f"[{session['session_name']}] built analyzer_beh via spikeinterface")
        else:
            log(f"[{session['session_name']}] analyzer_beh already present")

        # 2) analyzer_tag (if g1 session exists)
        if not session["analyzer_tag"].exists() and session["session_name"].endswith("_g0"):
            base_name = session["session_name"][:-3]
            sibling_g1 = session["session_dir"].parent / f"{base_name}_g1"
            g1_imec = list(sibling_g1.glob("*_imec0"))
            if sibling_g1.exists() and g1_imec:
                log(f"[{session['session_name']}] Found g1 at {sibling_g1}, creating analyzer_tag...")
                rec_g1 = si.read_spikeglx(str(g1_imec[0]), stream_name="imec0.ap")
                if rec_g1.get_num_segments() > 1:
                    lens = [rec_g1.get_num_samples(i) for i in range(rec_g1.get_num_segments())]
                    max_i = max(range(len(lens)), key=lambda i: lens[i])
                    rec_g1 = rec_g1.select_segments([max_i])
                
                # Apply same preprocessing to g1 recording
                rec_g1 = si.highpass_filter(rec_g1, freq_min=400.)
                bad_channel_ids_g1, _ = si.detect_bad_channels(rec_g1)
                if len(bad_channel_ids_g1) > 0:
                    rec_g1 = rec_g1.remove_channels(bad_channel_ids_g1)
                if "inter_sample_shift" in rec_g1.get_property_keys():
                    rec_g1 = si.phase_shift(rec_g1)
                else:
                    log(f"[{session['session_name']}] WARNING: 'inter_sample_shift' missing for g1. Skipping phase_shift.")
                rec_g1 = si.common_reference(rec_g1, operator="median", reference="global")

                # Shift sorting to g1 timeframe (assuming tcat = g0 + g1). This creates the "window" for the tagging session.
                g0_samples = rec_main.get_total_samples() - rec_g1.get_total_samples()
                log(f"[{session['session_name']}] Slicing sorting for tag analyzer: start_frame={g0_samples}, total={rec_main.get_total_samples()}")
                if g0_samples > 0:
                    sorting_g1 = sorting.frame_slice(start_frame=g0_samples, end_frame=rec_main.get_total_samples())
                    
                    # Slice the recording as well, so we use the tcat data for waveforms
                    rec_tag = rec_main.frame_slice(start_frame=g0_samples, end_frame=rec_main.get_total_samples())

                    # Filter spikes around stimulation events if available (window +/- 30ms)
                    sorting_to_use = sorting_g1
                    stim_path = session["ks_dir"] / "stimulation_seconds.npy"
                    if stim_path.exists():
                        import numpy as np
                        stim_times = np.load(stim_path)
                        fs = rec_main.get_sampling_frequency()

                        # Adjust stim times to g1 timeframe (tcat relative -> g1 relative)
                        g0_seconds = g0_samples / fs
                        stim_times_g1 = stim_times - g0_seconds

                        # Keep only stim times within g1 duration
                        g1_dur = rec_tag.get_total_duration()
                        stim_times_g1 = stim_times_g1[(stim_times_g1 >= 0) & (stim_times_g1 < g1_dur)]

                        if len(stim_times_g1) > 0:
                            # Use window from user snippet: -15ms to +25ms
                            pre_stim = -0.015
                            post_stim = 0.025
                            log(f"[{session['session_name']}] Filtering spikes around {len(stim_times_g1)} stim events ({pre_stim*1000}ms to +{post_stim*1000}ms)")
                            
                            new_spikes = {}
                            stim_times_g1.sort()
                            for uid in sorting_g1.get_unit_ids():
                                spikes = sorting_g1.get_unit_spike_train(uid)
                                spikes_sec = spikes / fs
                                
                                # Vectorized check: find spikes falling into any [t_stim+pre, t_stim+post] window
                                # We construct a mask of valid spikes
                                valid_mask = np.zeros(len(spikes), dtype=bool)
                                for t in stim_times_g1:
                                    valid_mask |= (spikes_sec >= t + pre_stim) & (spikes_sec <= t + post_stim)
                                
                                new_spikes[uid] = spikes[valid_mask]

                        sorting_to_use = si.NumpySorting.from_unit_dict(new_spikes, sampling_frequency=fs)

                    an_tag = si.create_sorting_analyzer(
                        sorting_to_use, rec_tag, format="binary_folder",
                        folder=str(session["analyzer_tag"]), overwrite=True,
                        sparse=True,
                        num_spikes_for_sparsity=100,
                        method="radius",
                        radius_um=100.0,
                        peak_sign="neg",
                        n_jobs=12,
                        chunk_duration="10s",
                        progress_bar=True
                    )
                    an_tag.compute("random_spikes")
                    an_tag.compute("waveforms", n_jobs=12, chunk_duration="10s", progress_bar=True)
                    an_tag.compute("templates", operators=["average", "median"])
                    log(f"[{session['session_name']}] built analyzer_tag via spikeinterface")
    finally:
        if created_temp_qmetrics and buggy_qmetrics_path and buggy_qmetrics_path.exists():
            log(f"[{session['session_name']}] Workaround: Cleaning up {buggy_qmetrics_path}")
            shutil.rmtree(buggy_qmetrics_path)


def preferred_nidq(session: Dict[str, Path]) -> Optional[Path]:
    sess_dir = session["session_dir"]
    # prefer tcat, otherwise fall back to t0
    for pattern in (f"{session['session_name']}_tcat.nidq.bin", "*tcat.nidq.bin", "*t0.nidq.bin", "*.nidq.bin"):
        found = list(sess_dir.glob(pattern))
        if found:
            return found[0]
    return None


def extract_digital_io(
    session: Dict[str, Path],
    tagging: bool,
    digital_lines: Optional[List[int]],
    stim_line: int,
    dry_run: bool,
    stim_protocol_cleanup: bool,
    stim_min_ibi: float,
    stim_max_ibi: float,
    stim_min_pulses: int,
) -> None:
    nidq_bin = preferred_nidq(session)
    if nidq_bin is None:
        log(f"[{session['session_name']}] no nidq.bin found, skipping digital extraction")
        return
    nidq_bin = Path(nidq_bin)

    ks_dir = session["ks_dir"]
    ks_dir.mkdir(parents=True, exist_ok=True)

    if (ks_dir / "spike_seconds.npy").exists():
        log(f"[{session['session_name']}] spike_seconds.npy already exists, skipping digital extraction")
        return

    try:
        import numpy as np  # type: ignore
        from DemoReadSGLXData.readSGLX import ExtractDigital, SampRate, makeMemMapRaw, readMeta  # type: ignore
    except Exception as exc:
        log(f"[{session['session_name']}] DemoReadSGLXData not available ({exc}), skipping digital extraction")
        return

    # Get NIDQ sampling rate
    nidq_meta = readMeta(nidq_bin)
    nidq_sRate = SampRate(nidq_meta)
    rawData = makeMemMapRaw(str(nidq_bin), nidq_meta)
    nChan, nFileSamp = rawData.shape

    dLineList = digital_lines if digital_lines else ([0, 1, 2, 4, stim_line] if tagging else [0])
    firstSamp = 0
    lastSamp = nFileSamp - 1
    digArray = ExtractDigital(rawData, firstSamp, lastSamp, 0, dLineList, nidq_meta)

    label_map = {0: "strobe", 1: "licking", 2: "reward", 4: "Trialstart", stim_line: "stimulation"}
    event_seconds: Dict[str, np.ndarray] = {}
    event_times: Dict[str, np.ndarray] = {}

    for idx, line in enumerate(dLineList):
        label = label_map.get(line, f"line{line}")
        arr = digArray[idx, :]
        
        times_on = np.where((arr[:-1] == 0) & (arr[1:] == 1))[0] + 1
        log(f"[{session['session_name']}] Detected {len(times_on)} rising edges on line {line} ('{label}')")

        if line == stim_line:
            secs_on = times_on / nidq_sRate
            times_off = np.where((arr[:-1] == 1) & (arr[1:] == 0))[0]
            secs_off = times_off / nidq_sRate

            if stim_protocol_cleanup and len(secs_on) >= stim_min_pulses:
                intervals = np.diff(secs_on)
                valid_start_idx = -1
                
                window_size = stim_min_pulses - 1
                if window_size > 0:
                    for i in range(len(intervals) - window_size + 1):
                        window = intervals[i : i + window_size]
                        if np.all((window >= stim_min_ibi) & (window <= stim_max_ibi)):
                            valid_start_idx = i
                            break
            
                if valid_start_idx != -1:
                    num_debris = valid_start_idx
                    log(f"[{session['session_name']}] Found stimulation protocol start. Trimming {num_debris} debris pulses.")
                    
                    times_on = times_on[valid_start_idx:]
                    secs_on = secs_on[valid_start_idx:]
                    
                    if len(times_off) > 0 and len(times_on) > 0:
                        first_valid_on_time = times_on[0]
                        first_off_idx = np.searchsorted(times_off, first_valid_on_time)
                        times_off = times_off[first_off_idx:]
                        secs_off = secs_off[first_off_idx:]
                else:
                    log(f"[{session['session_name']}] WARNING: Could not find stimulation protocol start pattern. Using all pulses.")
            
            event_seconds[label] = secs_on
            event_times[label] = times_on
            event_seconds[f"{label}_off"] = secs_off
            event_times[f"{label}_off"] = times_off
        else:
            secs_on = times_on / nidq_sRate
            event_seconds[label] = secs_on
            event_times[label] = times_on

    if dry_run:
        log(f"[{session['session_name']}] dry-run: digital events detected {list(event_seconds.keys())}")
    else:
        for label, arr in event_seconds.items():
            np.save(ks_dir / f"{label}_seconds.npy", arr)
        for label, arr in event_times.items():
            np.save(ks_dir / f"{label}_times.npy", arr)

    # Save spike_seconds from spike_times.npy for TPrime input, using the IMEC sampling rate
    spike_time_path = ks_dir / "spike_times.npy"
    if spike_time_path.exists():
        ap_meta_path = session.get("ap_meta")
        if ap_meta_path and ap_meta_path.exists():
            ap_meta = readMeta(ap_meta_path)
            imec_sRate = SampRate(ap_meta)
            spike_times_samples = np.load(spike_time_path)
            spike_seconds = spike_times_samples / imec_sRate
            if not dry_run:
                np.save(ks_dir / "spike_seconds.npy", spike_seconds)
            log(f"[{session['session_name']}] Converted spike_times to seconds using imec rate: {imec_sRate}")
        else:
            log(f"[{session['session_name']}] WARNING: AP meta file not found. Cannot convert spike_times to seconds.")
    else:
        log(f"[{session['session_name']}] spike_times.npy missing; skipping spike_seconds.npy creation")


def run_tprime(
    session: Dict[str, Path],
    tprime_exe: Optional[Path],
    sync_period: float,
    dry_run: bool,
) -> None:
    exe = tprime_exe or Path("TPrime")
    ni_edge = next(iter(session["session_dir"].glob("*nidq.xa_0_500.txt")), None)
    imec_edge = next(iter(session["imec_dir"].glob("*384_6_500.txt")), None)
    spike_seconds = session["ks_dir"] / "spike_seconds.npy"
    spike_seconds_adj = session["ks_dir"] / "spike_seconds_adj.npy"

    if spike_seconds_adj.exists():
        log(f"[{session['session_name']}] spike_seconds_adj.npy already exists, skipping TPrime")
        return

    if not ni_edge or not imec_edge or not spike_seconds.exists():
        log(f"[{session['session_name']}] missing inputs for TPrime; skipping")
        return

    cmd = [
        str(exe),
        f"-syncperiod={sync_period}",
        f"-tostream={normalize_path(ni_edge)}",
        f"-fromstream=1,{normalize_path(imec_edge)}",
        f"-events=1,{normalize_path(spike_seconds)},{normalize_path(spike_seconds_adj)}",
    ]
    run_cmd(cmd, dry_run=dry_run)


def build_spike_mask(
    session: Dict[str, Path],
    stim_line: int,
    tol: float,
    dry_run: bool,
) -> None:
    ks_dir = session["ks_dir"]
    stim = ks_dir / "stimulation_seconds.npy"
    stim_off = ks_dir / "stimulation_off_seconds.npy"
    spikes_adj = ks_dir / "spike_seconds_adj.npy"

    if (ks_dir / "spike_mask.npy").exists():
        log(f"[{session['session_name']}] spike_mask.npy already exists, skipping build_spike_mask")
        return

    if not (stim.exists() and stim_off.exists() and spikes_adj.exists()):
        log(f"[{session['session_name']}] missing stimulation/spike adj files; skipping spike_mask")
        return

    import numpy as np  # type: ignore

    spikes = np.load(spikes_adj)
    stim_times = np.concatenate([np.load(stim), np.load(stim_off)])
    mask = np.ones(len(spikes), dtype=bool)
    for t in stim_times:
        mask &= np.abs(spikes - t) > tol

    if dry_run:
        log(f"[{session['session_name']}] dry-run: would save spike_mask with {mask.sum()}/{len(mask)} kept")
    else:
        np.save(ks_dir / "spike_mask.npy", mask)


def export_waveforms_and_metrics(session: Dict[str, Path], dry_run: bool) -> None:
    ks_dir = session["ks_dir"]
    if not ks_dir.exists():
        log(f"[{session['session_name']}] missing kilosort4 folder, skip waveform export")
        return

    beh_folder = session["analyzer_beh"]
    if not beh_folder.exists():
        log(f"[{session['session_name']}] analyzer_beh not found, skip waveform export")
        return

    try:
        import numpy as np
        from spikeinterface_v2 import load_sorting_analyzer  # type: ignore
        from my_spike_tools import waveformPrepare_template  # type: ignore
    except ImportError as exc:
        log(f"[{session['session_name']}] spikeinterface_v2 not available ({exc}), skipping")
        return

    tag_folder = session["analyzer_tag"]
    has_tag = tag_folder.exists()

    log(f"[{session['session_name']}] loading analyzers")
    beh_an = load_sorting_analyzer(folder=str(beh_folder), format="binary_folder")
    tag_an = load_sorting_analyzer(folder=str(tag_folder), format="binary_folder") if has_tag else None

    # Ensure waveforms and median templates exist
    for an in [beh_an, tag_an]:
        if an is not None:
            if an.get_extension("random_spikes") is None:
                log(f"[{session['session_name']}] Computing random_spikes for {an.folder}")
                an.compute("random_spikes")

            if an.get_extension("waveforms") is None:
                log(f"[{session['session_name']}] Computing waveforms for {an.folder}")
                an.compute("waveforms")

            ext = an.get_extension("templates")
            if ext is None or "median" not in ext.data:
                log(f"[{session['session_name']}] Computing templates (including median) for {an.folder}")
                an.compute("templates", operators=["average", "median"])

    unit_ids = beh_an.sorting.unit_ids
    wf_beh_med: Dict[int, Optional[object]] = {}
    wf_beh_avg: Dict[int, Optional[object]] = {}
    wf_tag_med: Dict[int, Optional[object]] = {}
    wf_tag_avg: Dict[int, Optional[object]] = {}

    # Suppress noisy prints from my_spike_tools during waveform extraction
    with open(os.devnull, "w") as fnull:
        _orig_stdout = sys.stdout
        sys.stdout = fnull
        try:
            for uid in unit_ids:
                avg, med, _ = waveformPrepare_template(
                    beh_an, unit_id=uid, designated=False, use_template=True
                )
                wf_beh_avg[int(uid)] = avg
                wf_beh_med[int(uid)] = med

                if has_tag and tag_an is not None:
                    avg_t, med_t, _ = waveformPrepare_template(
                        tag_an, unit_id=uid, designated=False, use_template=False
                    )
                    wf_tag_avg[int(uid)] = avg_t
                    wf_tag_med[int(uid)] = med_t
        finally:
            sys.stdout = _orig_stdout

    log(f"[{session['session_name']}] computing template metrics/ACC/ISI")
    for ext in ("template_metrics", "correlograms", "isi_histograms", "unit_locations"):
        (beh_folder / "extensions" / ext).mkdir(parents=True, exist_ok=True)

    # Suppress noisy RuntimeWarnings/ConvergenceWarnings from scipy/sklearn during computation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tm = beh_an.compute(input="template_metrics", include_multi_channel_metrics=True, save=False)
        ccg = beh_an.compute(input="correlograms", window_ms=1000.0, bin_ms=5.0, method="auto", save=False)
        isi = beh_an.compute(input="isi_histograms", window_ms=1000.0, bin_ms=5.0, method="auto", save=False)
        ul = beh_an.compute(input="unit_locations", method="center_of_mass", save=False)

    unit_ids_list = list(unit_ids)
    ccg_dict = {int(uid): ccg.get_data()[0][i, i] for i, uid in enumerate(unit_ids_list)}
    isi_dict = {int(uid): isi.get_data()[0][i] for i, uid in enumerate(unit_ids_list)}

    if dry_run:
        log(f"[{session['session_name']}] dry-run: not writing metrics")
        return

    df_metrics = tm.get_data()
    locs = ul.get_data()
    df_metrics["x"] = locs[:, 0]
    df_metrics["y"] = locs[:, 1]
    df_metrics.index.name = "unit_id"
    df_metrics.to_csv(ks_dir / "template_metrics.csv", index=True)
    np.save(ks_dir / "ACC.npy", ccg_dict)
    np.save(ks_dir / "ISI.npy", isi_dict)
    np.save(ks_dir / "waveform_beh_average.npy", wf_beh_avg)
    np.save(ks_dir / "waveform_beh_median.npy", wf_beh_med)
    if has_tag:
        np.save(ks_dir / "waveform_tag_average.npy", wf_tag_avg)
        np.save(ks_dir / "waveform_tag_median.npy", wf_tag_med)
    log(f"[{session['session_name']}] waveform exports written to {ks_dir}")


def classify_cells(session: Dict[str, Path], dry_run: bool) -> None:
    ks_dir = session["ks_dir"]
    metrics_path = ks_dir / "template_metrics.csv"
    wf_path = ks_dir / "waveform_beh_average.npy"
    
    if (ks_dir / "unit_classification_rulebased.csv").exists():
        log(f"[{session['session_name']}] classification already exist, skipping classification")
        return
    if not metrics_path.exists() or not wf_path.exists():
        log(f"[{session['session_name']}] missing metrics or waveforms, skip classification")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        log(f"[{session['session_name']}] matplotlib/pandas not available ({exc}), skipping classify")
        return

    metrics = pd.read_csv(metrics_path)
    if "unit_id" in metrics.columns:
        metrics["unit_id"] = metrics["unit_id"].astype(int)
    else:
        # Fallback for older files or unnamed indices
        metrics["unit_id"] = metrics["Unnamed: 0"].astype(int)

    wf_dict = np.load(wf_path, allow_pickle=True).item()

    def classify_row(row: pd.Series) -> str:
        ptv = row["peak_to_valley"] * 1000
        hfw = row["half_width"] * 1000
        if ptv < 0.42 and hfw < 0.42:
            return "FSI"
        if ptv >= 0.42 and ptv < 1.5 and hfw < 0.75:
            return "MSN"
        return "Other"

    metrics["cell_type"] = metrics.apply(classify_row, axis=1)

    if dry_run:
        log(f"[{session['session_name']}] dry-run: not writing classification outputs")
        return

    colors = {"FSI": "C1", "MSN": "C0", "Other": "C2"}
    plt.figure(figsize=(8, 6))
    for ct in ("FSI", "MSN", "Other"):
        idx = metrics["cell_type"] == ct
        plt.scatter(
            metrics.loc[idx, "peak_to_valley"] * 1000,
            metrics.loc[idx, "half_width"] * 1000,
            c=colors[ct],
            label=ct,
            s=60,
            alpha=0.7,
        )
    plt.xlabel("Peak to Trough (ms)")
    plt.ylabel("Half Peak Width (ms)")
    plt.title("Cell Type Assignment (rule-based)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ks_dir / "rulebased_celltype_scatter.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    for ct in ("MSN", "FSI", "Other"):
        unit_ids = metrics.loc[metrics["cell_type"] == ct, "unit_id"]
        waveforms = [wf_dict[uid] for uid in unit_ids if uid in wf_dict and wf_dict[uid] is not None]
        if not waveforms:
            continue
        mean_wf = sum(waveforms) / len(waveforms)
        plt.plot(mean_wf, label=ct)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Average Waveform by Cell Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ks_dir / "rulebased_celltype_waveforms.png")
    plt.close()

    out_csv = ks_dir / "unit_classification_rulebased.csv"
    metrics[["unit_id", "peak_to_valley", "half_width", "cell_type"]].to_csv(out_csv, index=False)
    log(f"[{session['session_name']}] classification saved to {out_csv}")


def run_lfp_csd(session: Dict[str, Path], dry_run: bool) -> None:
    imec_dir = session["imec_dir"]
    lfp_dir = imec_dir / "LFP"
    if lfp_dir.exists():
        log(f"[{session['session_name']}] LFP already exists, skipping")
        return

    log(f"[{session['session_name']}] Extracting LFP to {lfp_dir}")
    if dry_run:
        return

    try:
        import spikeinterface.full as si
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import welch
    except ImportError as e:
        log(f"[{session['session_name']}] SpikeInterface, matplotlib, or scipy not found: {e}")
        return

    # Load recording (use original to ensure metadata like inter_sample_shift is correct)
    try:
        rec = si.read_spikeglx(str(imec_dir), stream_name="imec0.ap")
    except Exception as e:
        log(f"[{session['session_name']}] Error loading SpikeGLX recording for LFP: {e}")
        return

    if rec.get_num_segments() > 1:
        lens = [rec.get_num_samples(i) for i in range(rec.get_num_segments())]
        max_i = max(range(len(lens)), key=lambda i: lens[i])
        rec = rec.select_segments([max_i])

    # Preserve raw recording for gain check (before any processing)
    rec_raw = rec

    # Preprocessing pipeline
    # 1. Phase shift (if available)
    if "inter_sample_shift" in rec.get_property_keys():
        log(f"[{session['session_name']}] Applying phase shift")
        rec = si.phase_shift(rec)
    else:
        log(f"[{session['session_name']}] 'inter_sample_shift' not found, skipping phase shift")
    
    # 2. Common Median Reference
    log(f"[{session['session_name']}] Applying Common Median Reference")
    rec = si.common_reference(rec, operator="median", reference="global")
    lfp_dir.mkdir(parents=True, exist_ok=True)
    rec_30s = rec.frame_slice(start_frame=0, end_frame=int(30 * rec.get_sampling_frequency()))
    rec_30s.save(folder=lfp_dir / "lfp_30s_cmr", overwrite=True, n_jobs=12, chunk_duration="2s", max_memory_per_bundle="4G", progress_bar=True)

    # 3 Remove Large "Surges" (Broadband Glitches)
    try:
        surge_triggers = _detect_large_artifacts(rec_raw, session, z_threshold=10.0, chunk_dur=1.0)
        if len(surge_triggers) > 0:
            log(f"[{session['session_name']}] Removing {len(surge_triggers)} large artifact events (surges)")
            # Remove 4ms window around surge: 2ms before, 2ms after
            rec = si.remove_artifacts(
                rec, 
                list_triggers=surge_triggers, 
                ms_before=2.0, 
                ms_after=2.0, 
                mode='cubic'
            )
        else:
            log(f"[{session['session_name']}] No large surges detected.")
    except Exception as e:
        log(f"[{session['session_name']}] WARNING: Surge detection/removal failed: {e}")
    
    # 4 Resample to 1000 Hz (includes anti-aliasing)
    log(f"[{session['session_name']}] Resampling to 1000 Hz")
    rec = si.resample(rec, resample_rate=1000)

    # 5. Bandpass filter (0.5 - 200 Hz)
    log(f"[{session['session_name']}] Bandpass filtering (0.5-200 Hz)")
    rec = si.bandpass_filter(rec, freq_min=0.5, freq_max=200.0, margin_ms=100.0)

    # 6. Save LFP
    lfp_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{session['session_name']}] Saving LFP to {lfp_dir}")
    rec.save(folder=lfp_dir, overwrite=True, n_jobs=12, chunk_duration="2s", max_memory_per_bundle="4G",progress_bar=True)
    
    # --- Sanity Check Plots ---
    log(f"[{session['session_name']}] Generating sanity check plots")
    
    fs = rec.get_sampling_frequency()
    total_samples = rec.get_total_samples()
    total_dur = total_samples / fs
    
    # 1. Identify shanks and group channels
    locations = rec.get_channel_locations()
    x_coords = locations[:, 0]
    y_coords = locations[:, 1]
    
    # Group x-coordinates into 4 shanks (e.g. 0, 250, 500, 750)
    # Using floor division by 200 provides a robust grouping for NP 2.0 4-shank probes
    shank_ids = (x_coords // 200).astype(int)
    unique_shanks = np.unique(shank_ids)
    num_shanks = len(unique_shanks)
    
    # Calculate global RMS
    rms_dur = min(10.0, total_dur)
    rms_samples = int(rms_dur * fs)
    rms_start = int((total_dur/2 - rms_dur/2) * fs)
    traces_rms = rec.get_traces(start_frame=rms_start, end_frame=rms_start+rms_samples)
    rms_per_chan = np.sqrt(np.mean(traces_rms**2, axis=0))

    # 2s snapshot for heatmaps and traces
    snap_dur = 2.0
    snap_samples = int(snap_dur * fs)
    snap_start = int(total_dur/2 * fs)
    traces_snap_full = rec.get_traces(start_frame=snap_start, end_frame=snap_start+snap_samples)
    time_snap = np.arange(snap_samples) / fs
    
    # PSD (60s window)
    psd_dur = 60
    psd_samples = int(psd_dur * fs)
    psd_start = max(0, int(snap_start - psd_samples/2))
    psd_end = min(int(total_samples), psd_start + psd_samples)
    traces_psd_full = rec.get_traces(start_frame=psd_start, end_frame=psd_end)
    f_psd, Pxx_full = welch(traces_psd_full, fs=fs, nperseg=2048, axis=0)
    f_mask = f_psd <= 150
    f_psd = f_psd[f_mask]
    Pxx_full_db = 10 * np.log10(Pxx_full[f_mask, :])

    # Plotting setup
    # Rows: 0=Heatmap, 1=PSD, 2=Traces
    # Cols: 0..num_shanks-1 = Shanks, last col = RMS Summary
    fig = plt.figure(figsize=(24, 18))
    # Standard shank width ratios: 1 for each shank, 1.2 for the RMS summary on the right
    w_ratios = [1.0] * num_shanks + [1.2]
    gs = fig.add_gridspec(3, num_shanks + 1, height_ratios=[1, 1, 1], width_ratios=w_ratios)
    
    vmax_lfp = np.percentile(np.abs(traces_snap_full), 98)
    
    for i, sid in enumerate(unique_shanks):
        # Channels on this shank
        s_idx = np.where(shank_ids == sid)[0]
        # Sort by depth
        s_y = y_coords[s_idx]
        s_sort = np.argsort(s_y)[::-1]
        orig_indices = s_idx[s_sort]
        s_depths = s_y[s_sort]
        
        # A. LFP Heatmap
        ax_lfp = fig.add_subplot(gs[0, i])
        s_traces = traces_snap_full[:, orig_indices]
        ax_lfp.imshow(s_traces.T, aspect='auto', cmap='RdBu_r', 
                      extent=[0, snap_dur, s_depths.min(), s_depths.max()],
                      vmax=vmax_lfp, vmin=-vmax_lfp)
        ax_lfp.set_title(f"Shank {i} (Heatmap)")
        if i == 0: ax_lfp.set_ylabel("Depth (um)")
        ax_lfp.set_xlabel("Time (s)")

        # B. PSD Heatmap
        ax_psd = fig.add_subplot(gs[1, i])
        s_psd = Pxx_full_db[:, orig_indices]
        ax_psd.imshow(s_psd.T, aspect='auto', cmap='viridis',
                      extent=[f_psd.min(), f_psd.max(), s_depths.min(), s_depths.max()])
        ax_psd.set_title(f"Shank {i} (PSD)")
        if i == 0: ax_psd.set_ylabel("Depth (um)")
        ax_psd.set_xlabel("Freq (Hz)")

        # C. Raw Traces (Subset for clarity)
        ax_tr = fig.add_subplot(gs[2, i])
        sel_step = max(1, len(s_depths) // 40)
        sel_idx = np.arange(0, len(s_depths), sel_step)
        offset = np.std(s_traces) * 15
        for j, s_ptr in enumerate(sel_idx):
            ax_tr.plot(time_snap, s_traces[:, s_ptr] + j * offset, color='k', lw=0.3)
        ax_tr.set_title(f"Shank {i} (Traces)")
        ax_tr.set_yticks(np.arange(len(sel_idx)) * offset)
        ax_tr.set_yticklabels([f"{int(d)}um" for d in s_depths[sel_idx]])
        if i == 0: ax_tr.set_ylabel("Channel (Depth)")
        ax_tr.set_xlabel("Time (s)")

    # D. RMS summary plot (spanning all rows on right)
    ax_rms = fig.add_subplot(gs[:, num_shanks])
    for i, sid in enumerate(unique_shanks):
        s_idx = np.where(shank_ids == sid)[0]
        s_y = y_coords[s_idx]
        s_sort = np.argsort(s_y)
        ax_rms.plot(rms_per_chan[s_idx[s_sort]], s_y[s_sort], label=f"Shank {i}")
    
    ax_rms.set_title("RMS per Channel (All Shanks)")
    ax_rms.set_xlabel("RMS (uV)")
    ax_rms.set_ylabel("Depth (um)")
    ax_rms.legend()
    ax_rms.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(lfp_dir / "lfp_sanity_check.png", dpi=150)
    plt.close(fig)
    
    log(f"[{session['session_name']}] Refined multishank LFP plots saved to {lfp_dir / 'lfp_sanity_check.png'}")
    
    # --- 10-bit ADC Gain Check ---
    # Use rec_raw to ensure we check the absolute raw integers (no phase shift/interpolation)
    _plot_gain_check(session, rec_raw, lfp_dir, fs, total_dur)
    
    log(f"[{session['session_name']}] LFP extraction and sanity check complete.")

def _detect_large_artifacts(rec, session, z_threshold=8.0, chunk_dur=1.0, verbose=True):
    """
    Detects high-amplitude broadband artifacts using robust statistics (MAD).
    Returns a list of trigger frames (centers of artifacts).
    """
    fs = rec.get_sampling_frequency()
    n_samples = rec.get_total_samples()
    n_channels = rec.get_num_channels()
    
    # 1. Estimate noise levels (MAD) from a few random chunks
    n_noise_chunks = 5
    starts = np.linspace(0, n_samples - fs * chunk_dur, n_noise_chunks).astype(int)
    
    mads = []
    medians = []
    
    if verbose: log(f"[{session['session_name']}] Estimating noise levels for artifact detection...")
    
    for start in starts:
        end = int(start + fs * chunk_dur)
        traces = rec.get_traces(start_frame=start, end_frame=end, return_scaled=False)
        if traces is None: continue
        # Robust std estim: MAD / 0.6745
        # We compute per channel
        tr_med = np.median(traces, axis=0)
        tr_mad = np.median(np.abs(traces - tr_med), axis=0) / 0.6745
        mads.append(tr_mad)
        medians.append(tr_med)
        
    global_mad = np.median(np.vstack(mads), axis=0)
    global_med = np.median(np.vstack(medians), axis=0)
    
    # 2. Scan file for outliers
    chunk_size = int(60.0 * fs) # 60 seconds scan (larger chunks for speed)
    artifact_triggers = []
    
    # Strategy: Compute median trace across a logical SUBSET of channels (CMR trace)
    start_f = 0
    if verbose: log(f"[{session['session_name']}] Scanning for large artifacts (Threshold: {z_threshold} sigma)...")
    
    skip_ch = 8
    sel_chan_ids = rec.channel_ids[::skip_ch]
    
    # Create ranges for the progress bar
    starts = range(0, n_samples, chunk_size)
    
    for start_f in tqdm(starts, desc="Detecting Surges", unit="chunk"):
        end_f = min(start_f + chunk_size, n_samples)
        # Read only subset of channels
        traces = rec.get_traces(start_frame=start_f, end_frame=end_f, channel_ids=sel_chan_ids, return_scaled=False)
        
        # Common Mode (median across channels at each timepoint)
        # shape: (n_time,)
        cm_trace = np.median(traces, axis=1)
        
        # Robust stats for CM trace
        cm_med = np.median(cm_trace)
        cm_mad = np.median(np.abs(cm_trace - cm_med)) / 0.6745
        
        # Detect
        # We use a high threshold on the Common Mode signal
        mask = np.abs(cm_trace - cm_med) > (z_threshold * cm_mad)
        
        if np.any(mask):
            # Find indices
            bad_indices = np.where(mask)[0] + start_f
            artifact_triggers.append(bad_indices)
        
    if len(artifact_triggers) > 0:
        all_bad = np.concatenate(artifact_triggers)
        all_bad = np.unique(all_bad)
        diffs = np.diff(all_bad)
        # Simple clustering: if gap > 10ms (300 samples), it's a new event
        split_indices = np.where(diffs > 300)[0] + 1
        events = np.split(all_bad, split_indices)
        
        # Take center of each event
        centers = [int(np.median(e)) for e in events]
        return centers
    
    return []

def _plot_gain_check(session, rec, lfp_dir, fs, total_dur):
    """
    Generates a histogram of raw int16 data to check for gain saturation/clipping.
    Uses chunked loading to cover 500s of data (200 chunks x 2.5s).
    """
    log(f"[{session['session_name']}] generating gain check plot (500s sample)...")
    
    import matplotlib.pyplot as plt
    import numpy as np

    # Parameters
    n_chunks = 200
    chunk_dur = 2.5 # seconds
    
    total_samples = rec.get_total_samples()
    chunk_samples = int(chunk_dur * fs)
    
    req_dur = n_chunks * chunk_dur
    if total_dur <= req_dur:
        starts = np.arange(0, total_samples, chunk_samples)
    else:
        # Evenly spaced starts
        starts = np.linspace(0, total_samples - chunk_samples, n_chunks).astype(int)

    # Accumulate histogram
    # int16 range: -8192 8191
    bins = np.linspace(-8192, 8191, 1000) # 1000 bins for overview
    counts = np.zeros(len(bins)-1, dtype=np.int64)
    
    min_val, max_val = 8191, -8192
    total_clipped_14 = 0
    total_points = 0
    
    for start_frame in starts:
        end_frame = min(total_samples, start_frame + chunk_samples)
        if start_frame >= end_frame: continue
        
        # Get raw traces (int16)
        chunk = rec.get_traces(start_frame=int(start_frame), end_frame=int(end_frame), return_scaled=False)
        if chunk is None or chunk.size == 0: continue
        
        # Flatten
        vals = chunk.flatten()
        
        # Update histograms
        c, _ = np.histogram(vals, bins=bins)
        counts += c
        
        # Update stats
        if len(vals) > 0:
            min_val = min(min_val, vals.min())
            max_val = max(max_val, vals.max())
        
        # 14-bit limits (approx +/- 8192)
        # Check if values hit 14-bit rails (implies clipping if raw ADC is 14-bit)
        clipped_mask_14 = (vals <= -8192) | (vals >= 8191)
        total_clipped_14 += np.sum(clipped_mask_14)
        
        total_points += len(vals)

    # Diagnostic: Check for quantization/scaling
    unique_vals = np.unique(vals)
    if len(unique_vals) > 1:
        steps = np.diff(unique_vals)
        raw_step = float(np.median(steps)) if len(steps) > 0 else 1.0
    else:
        raw_step = 1.0
        
    log(f"[{session['session_name']}] Gain Check: Min={min_val}, Max={max_val}, Detected Step Size={raw_step}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Center bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Log scale y-axis to see tails
    ax.bar(bin_centers, counts, width=np.diff(bins), color='steelblue', log=True, edgecolor='none')
    
    title_str = (f"Raw Signal Distribution (Gain Check)\n"
                 f"Session: {session['session_name']} | {n_chunks}x{chunk_dur}s chunks\n"
                 f"Detected Quantization Step: {raw_step} (Effective Limits might differ)")
    ax.set_title(title_str)
    ax.set_xlabel("Raw Value (int16)")
    ax.set_ylabel("Count (Log Scale)")
    ax.grid(True, which="both", alpha=0.3)
    
    # Secondary x-axis for uV
    try:
        gains = rec.get_channel_gains()
        med_gain = np.median(gains) if gains is not None else 1.0
    except:
        med_gain = 1.0
        
    def raw_to_uv(x):
        return x * med_gain
        
    def uv_to_raw(x):
        return x / med_gain
        
    secax = ax.secondary_xaxis('top', functions=(raw_to_uv, uv_to_raw))
    secax.set_xlabel('Voltage (uV)')
    
    # Vertical lines for 14-bit limits (Neuropixels 2.0 / Wideband)
    # 14-bit signed range: -8192 to 8191
    ax.axvline(-8192, color='orange', linestyle='--', linewidth=2, label='14-bit Limit')
    ax.axvline(8191, color='orange', linestyle='--', linewidth=2)
    
    # Legend
    ax.legend(loc='upper right')
    
    # Annotation focusing on 14-bit check
    clip_pct_14 = (total_clipped_14 / total_points) * 100 if total_points > 0 else 0
    
    stats_text = (
        f"Total Samples: {total_points/1e6:.1f} M\n"
        f"Min Value: {min_val} ({min_val*med_gain:.1f} uV)\n"
        f"Max Value: {max_val} ({max_val*med_gain:.1f} uV)\n"
        f"Est. Gain: {med_gain:.3f} uV/bit\n"
        f"Exceeds 14-bit: {total_clipped_14} ({clip_pct_14:.4f}%)"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    fig.savefig(lfp_dir / "gain_check.png", dpi=150)
    plt.close(fig)
    log(f"[{session['session_name']}] gain_check.png saved.")

def get_rastermap_subsets(session: Dict[str, Path]) -> Dict[str, List[int]]:
    """
    Returns a dictionary mapping subset names to lists of unit IDs.
    """
    ks_dir = session["ks_dir"]
    imec_dir = session["imec_dir"]
    
    # 1. Load Quality Labels (BombCell)
    # Expected path: imec_dir / kilosort4qMetrics / templates._bc_unit_labels.tsv
    bc_labels_path = imec_dir / "kilosort4qMetrics" / "templates._bc_unit_labels.tsv"
    
    # 2. Load Cell Types (Rule-based)
    # Expected path: ks_dir / unit_classification_rulebased.csv
    ct_path = ks_dir / "unit_classification_rulebased.csv"
    
    try:
        import pandas as pd
    except ImportError:
        return {}

    good_mua_units = []
    if bc_labels_path.exists():
        df_bc = pd.read_csv(bc_labels_path, sep="\t")
        
        # 1. Determine Unit ID column
        # BombCell might use 'unit_id', 'cluster_id', 'clusterID', or 'template_id'
        id_col = next((c for c in df_bc.columns if c.lower() in ("unit_id", "cluster_id", "clusterid", "template_id")), None)
        
        # 2. Determine Quality Mask
        # User specified: 1 is good, 2 is MUA, 3 is non-somatic, 0 is noise.
        # Fallback to string-based if numeric fails.
        if "unitType" in df_bc.columns:
            ut = df_bc["unitType"]
            # Try numeric first
            mask = ut.isin([1, 2])
            if not mask.any():
                # Try string-based just in case
                mask = ut.astype(str).str.lower().isin(["good", "mua", "1", "2", "1.0", "2.0"])
        else:
            log(f"[{session['session_name']}] WARNING: 'unitType' column not found in {bc_labels_path}. Available: {list(df_bc.columns)}")
            mask = pd.Series([False] * len(df_bc))

        if id_col and mask.any():
            good_mua_units = df_bc.loc[mask, id_col].astype(int).tolist()
        elif not id_col and mask.any():
            log(f"[{session['session_name']}] WARNING: No unit ID column found in {bc_labels_path}. Assuming row index corresponds to unit ID.")
            good_mua_units = df_bc.index[mask].tolist()
        elif not id_col and not mask.any():
             log(f"[{session['session_name']}] WARNING: Could not find unit ID column in {bc_labels_path} and no units match quality filter. Available: {list(df_bc.columns)}")
    else:
        log(f"[{session['session_name']}] WARNING: BombCell labels not found at {bc_labels_path}. Using all units for Quality filter.")
        # If BombCell is missing, we might have to fall back to 'good' from Kilosort if possible,
        # but the request implies BombCell is part of the flow. 
        # For now, if missing, we'll return empty or log.
        return {}

    msn_units = []
    fsi_units = []
    if ct_path.exists():
        df_ct = pd.read_csv(ct_path)
        msn_units = df_ct.loc[df_ct["cell_type"] == "MSN", "unit_id"].astype(int).tolist()
        fsi_units = df_ct.loc[df_ct["cell_type"] == "FSI", "unit_id"].astype(int).tolist()
    else:
        log(f"[{session['session_name']}] WARNING: Cell classification not found at {ct_path}.")

    # Intersect with Good/MUA
    good_mua_set = set(good_mua_units)
    msn_in_good = [u for u in msn_units if u in good_mua_set]
    fsi_in_good = [u for u in fsi_units if u in good_mua_set]
    
    subsets = {
        "all_good_mua": good_mua_units,
        "msn_fsi": msn_in_good + fsi_in_good,
        "msn_only": msn_in_good
    }
    
    return subsets


def run_rastermap(
    session: Dict[str, Path], 
    dry_run: bool, 
    subset_name: str,
    target_unit_ids: List[int],
    bin_size_ms: float = 50.0
) -> None:
    """
    Runs Rastermap on a subset of units and saves results in imec_dir / rastermap / subset_name.
    """
    ks_dir = session["ks_dir"]
    imec_dir = session["imec_dir"]
    rastermap_dir = imec_dir / "rastermap" / subset_name

    if not target_unit_ids:
        log(f"[{session['session_name']}] No units for rastermap subset '{subset_name}', skipping")
        return

    if rastermap_dir.exists() and any(rastermap_dir.iterdir()):
        log(f"[{session['session_name']}] rastermap/{subset_name} folder already present, skipping")
        return

    spike_times_path = ks_dir / "spike_times.npy"
    spike_clusters_path = ks_dir / "spike_clusters.npy"
    
    if not (spike_times_path.exists() and spike_clusters_path.exists()):
        log(f"[{session['session_name']}] missing spike_times or spike_clusters; skipping rastermap")
        return

    if dry_run:
        log(f"[{session['session_name']}] dry-run: would run rastermap ({len(target_unit_ids)} units) and save to {rastermap_dir}")
        return

    try:
        import numpy as np
        from rastermap import Rastermap
    except ImportError as exc:
        log(f"[{session['session_name']}] rastermap or numpy not available ({exc}), skipping")
        return

    log(f"[{session['session_name']}] Loading spikes for rastermap subset '{subset_name}'...")
    all_spike_times = np.load(spike_times_path)
    all_spike_clusters = np.load(spike_clusters_path)

    # Filter to target units
    target_set = set(target_unit_ids)
    mask = np.isin(all_spike_clusters, target_unit_ids)
    spike_times = all_spike_times[mask]
    spike_clusters = all_spike_clusters[mask]

    if len(spike_times) == 0:
         log(f"[{session['session_name']}] subset '{subset_name}' has 0 spikes, skipping")
         return

    # Get sampling rate from meta to convert samples to seconds
    ap_meta_path = session.get("ap_meta")
    if not ap_meta_path or not ap_meta_path.exists():
        log(f"[{session['session_name']}] missing ap_meta; cannot determine sampling rate for rastermap")
        return
    
    try:
        from DemoReadSGLXData.readSGLX import readMeta, SampRate
        ap_meta = readMeta(ap_meta_path)
        fs = SampRate(ap_meta)
    except Exception as e:
        log(f"[{session['session_name']}] error reading meta for rastermap: {e}")
        return

    spike_seconds = spike_times / fs
    duration = spike_seconds.max()
    
    bin_size = bin_size_ms / 1000.0
    n_bins = int(np.ceil(duration / bin_size))
    
    unique_clusters = np.array(target_unit_ids)
    n_neurons = len(unique_clusters)
    
    log(f"[{session['session_name']}] Binning {n_neurons} clusters into {n_bins} bins ({bin_size_ms}ms)...")
    
    # Create binned matrix
    # X: (n_neurons, n_bins)
    X = np.zeros((n_neurons, n_bins), dtype=np.float32)
    
    # Map cluster IDs to 0-indexed indices
    cluster_to_idx = {cid: i for i, cid in enumerate(unique_clusters)}
    
    for cid in unique_clusters:
        idx = cluster_to_idx[cid]
        c_spikes = spike_seconds[spike_clusters == cid]
        if len(c_spikes) == 0: continue
        bins = (c_spikes / bin_size).astype(int)
        bins = bins[bins < n_bins]
        # Use np.add.at for efficient counting
        np.add.at(X[idx], bins, 1)

    log(f"[{session['session_name']}] Running Rastermap on subset '{subset_name}'...")
    model = Rastermap(locality=0.75, time_lag_window=15)
    model.fit(X)
    
    rastermap_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    np.save(rastermap_dir / "embedding.npy", model.embedding) # (n_neurons, 1)
    np.save(rastermap_dir / "isort.npy", model.isort)         # (n_neurons,) sorting index
    np.save(rastermap_dir / "X_binned.npy", X)               # Save binned data too
    np.save(rastermap_dir / "cluster_ids.npy", unique_clusters)
    
    log(f"[{session['session_name']}] Rastermap results for '{subset_name}' saved to {rastermap_dir}")

    # Generate and save a visual heatmap
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import zscore
        
        # Reorder activity by embedding
        X_sort = X[model.isort]
        
        # Z-score for visualization (per neuron)
        # Handle NaNs from constant/zero firing neurons
        X_z = zscore(X_sort, axis=1)
        X_z = np.nan_to_num(X_z) 

        plt.figure(figsize=(12, 8))
        # Use grayscale inverted (white background) and limit range for contrast
        plt.imshow(X_z, aspect='auto', cmap='gray_r', vmin=0, vmax=3.5)
        plt.colorbar(label='Z-scored spike count')
        plt.title(f"Rastermap Sorted Activity: {subset_name} (bin={bin_size_ms}ms)")
        plt.xlabel("Time (bins)")
        plt.ylabel("Neurons (sorted by embedding)")
        
        plot_path = rastermap_dir / "rastermap_heatmap.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        log(f"[{session['session_name']}] Rastermap heatmap saved to {plot_path}")

        # 2. Sorted Raster Plot (individual spikes)
        # Map each neuron's index in X to its rank in the sorting
        inverse_isort = np.zeros(n_neurons, dtype=int)
        inverse_isort[model.isort] = np.arange(n_neurons)

        plt.figure(figsize=(15, 10))
        
        # Calculate y-positions for each spike
        # Map cluster IDs to their index in X, then to their rank in isort
        ranks = np.array([inverse_isort[cluster_to_idx[cid]] for cid in spike_clusters])
        
        # Plot with small markers for large datasets
        plt.scatter(spike_seconds, ranks, s=0.5, c='black', marker='|', linewidths=0.5, alpha=0.5)
        
        plt.title(f"Rastermap Sorted Raster: {subset_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Neurons (sorted by embedding)")
        plt.tight_layout()
        
        raster_path = rastermap_dir / "rastermap_raster.png"
        plt.savefig(raster_path, dpi=200, bbox_inches='tight')
        plt.close()
        log(f"[{session['session_name']}] Rastermap sorted raster saved to {raster_path}")

        # 3. Spatial Probe Plot
        unit_x, unit_y = None, None
        valid_mask = None
        
        # Priority 1: spike_positions.npy (Raw Kilosort output)
        spike_positions_path = ks_dir / "spike_positions.npy"
        if spike_positions_path.exists():
            try:
                # We need to filter positions the same way as clusters
                all_spike_positions = np.load(spike_positions_path)
                # mask was computed earlier: mask = np.isin(all_spike_clusters, target_unit_ids)
                subset_positions = all_spike_positions[mask]
                
                unit_x = []
                unit_y = []
                valid_mask = []
                for cid in unique_clusters:
                    c_pos = subset_positions[spike_clusters == cid]
                    if len(c_pos) > 0:
                        med_pos = np.median(c_pos, axis=0)
                        unit_x.append(med_pos[0])
                        unit_y.append(med_pos[1])
                        valid_mask.append(True)
                    else:
                        unit_x.append(np.nan)
                        unit_y.append(np.nan)
                        valid_mask.append(False)
                
                unit_x = np.array(unit_x)
                unit_y = np.array(unit_y)
                valid_mask = np.array(valid_mask)
                log(f"[{session['session_name']}] Computed unit locations from spike_positions.npy")
            except Exception as e:
                log(f"[{session['session_name']}] WARNING: Failed to compute locations from spike_positions: {e}")

        # Priority 2: template_metrics.csv
        if unit_x is None or not valid_mask.any():
            metrics_path = ks_dir / "template_metrics.csv"
            if metrics_path.exists():
                import pandas as pd
                df_m = pd.read_csv(metrics_path)
                if "unit_id" in df_m.columns:
                    df_m = df_m.set_index("unit_id")
                
                if 'x' in df_m.columns and 'y' in df_m.columns:
                    df_subset = df_m.reindex(unique_clusters)
                    unit_x = df_subset['x'].values
                    unit_y = df_subset['y'].values
                    valid_mask = ~np.isnan(unit_x) & ~np.isnan(unit_y)
                    log(f"[{session['session_name']}] Loaded unit locations from template_metrics.csv")

        # Priority 3: SpikeInterface Analyzer
        if unit_x is None or not valid_mask.any():
            beh_folder = imec_dir / "analyzer_beh"
            if beh_folder.exists():
                try:
                    from spikeinterface_v2 import load_sorting_analyzer
                    beh_an = load_sorting_analyzer(folder=str(beh_folder), format="binary_folder")
                    ul = beh_an.compute("unit_locations", method="center_of_mass")
                    locs = ul.get_data()
                    all_unit_ids = beh_an.unit_ids
                    id_to_idx = {uid: i for i, uid in enumerate(all_unit_ids)}
                    
                    unit_x, unit_y, valid_mask = [], [], []
                    for cid in unique_clusters:
                        if cid in id_to_idx:
                            i = id_to_idx[cid]
                            unit_x.append(locs[i, 0]); unit_y.append(locs[i, 1]); valid_mask.append(True)
                        else:
                            unit_x.append(np.nan); unit_y.append(np.nan); valid_mask.append(False)
                    
                    unit_x = np.array(unit_x); unit_y = np.array(unit_y); valid_mask = np.array(valid_mask)
                    log(f"[{session['session_name']}] Computed unit locations from SpikeInterface analyzer")
                except Exception as e:
                    log(f"[{session['session_name']}] WARNING: Failed to compute from analyzer: {e}")

        # Final: Plot if we found anything
        if unit_x is not None and valid_mask.any():
            try:
                plt.figure(figsize=(10, 12))
                shank_ids = (unit_x // 200).astype(float)
                unique_shanks_present = np.unique(shank_ids[valid_mask])
                cmap = plt.get_cmap('gist_rainbow')
                emb = model.embedding.flatten()
                emb_norm = (emb - emb.min()) / (emb.max() - emb.min() + 1e-8)

                for s_id in unique_shanks_present:
                    s_mask = (shank_ids == s_id) & valid_mask
                    plt.scatter(unit_x[s_mask], unit_y[s_mask], 
                                c=emb_norm[s_mask], cmap=cmap, 
                                s=100, edgecolors='k', alpha=0.8)
                
                plt.colorbar(label='Rastermap Embedding (normalized)')
                plt.title(f"Spatial Neuron Distribution: {subset_name}")
                plt.xlabel("X Position (um)")
                plt.ylabel("Y Position (um)")
                if len(unique_shanks_present) > 1: plt.xlim(-50, 800)
                
                spatial_path = rastermap_dir / "rastermap_spatial.png"
                plt.savefig(spatial_path, dpi=150, bbox_inches='tight'); plt.close()
                log(f"[{session['session_name']}] Rastermap spatial probe plot saved to {spatial_path}")
            except Exception as e:
                log(f"[{session['session_name']}] WARNING: Error during spatial plotting: {e}")
        else:
            log(f"[{session['session_name']}] WARNING: No unit coordinates found from any source.")

    except Exception as e:
        log(f"[{session['session_name']}] WARNING: Could not generate Rastermap plots: {e}")


def parse_steps(step_arg: str) -> List[str]:
    all_steps = [
        "catgt",
        "channelmap",
        "kilosort",
        "digital",
        "tprime",
        "mask",
        "bombcell",
        "spikeinterface",
        "waveform_metrics",
        "classify",
        "lfp_csd",
        "rastermap",
    ]
    if step_arg.lower() == "all":
        return all_steps
    requested = [s.strip().lower() for s in step_arg.split(",") if s.strip()]
    bad = [s for s in requested if s not in all_steps]
    if bad:
        raise ValueError(f"Unknown step(s): {', '.join(bad)}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neuropixels pipeline across sessions")
    parser.add_argument("--data-root", required=True, nargs="+", help="Folder(s) containing session subfolders")
    parser.add_argument("--sessions", nargs="*", help="Optional explicit session folder names under data-root(s)")
    parser.add_argument("--steps", default="all", help="Comma list of steps or 'all'")
    parser.add_argument("--matlab", default="matlab", help="MATLAB executable for SGLXMetaToCoords/BombCell")
    parser.add_argument("--catgt-dir", type=Path, help="Directory containing runit.bat (CatGT); used as cwd")
    parser.add_argument("--bombcell-path", type=Path, default=REPO_ROOT / "Preprocessing", help="Root folder of BombCell (for addpath)")
    parser.add_argument(
        "--catgt-template",
        default=None,
        help="Command template for CatGT (e.g., \"runit.bat -dir={data_root} -run={session_name} -g=0,1 -t=0 -prb_fld -prb=0 -ni -ap\").",
    )
    parser.add_argument(
        "--kilosort-template",
        default=None,
        help="Command template for Kilosort. Placeholders: {python} {session_name} {session_dir} {imec_dir} {ap_bin} {ap_meta} {ks_dir}",
    )
    parser.add_argument( 
        "--kilosort-output-subdir",
        default="sorter_output",
        help="Subdirectory within the kilosort4 folder containing the raw sorter output. Default: sorter_output",
    )
    parser.add_argument(
        "--no-spikeinterface-kilosort",
        action="store_false",
        dest="use_spikeinterface_kilosort",
        default=True,
        help="Disable spikeinterface fallback for Kilosort if you prefer to provide a template or run manually. Enabled by default unless you pass this flag.",
    )
    parser.add_argument(
        "--spikeinterface-template",
        default=None,
        help="Command template to build analyzer_beh/analyzer_tag (if you prefer running your own script). "
             "Placeholders: {python} {session_name} {session_dir} {imec_dir} {ap_bin} {ap_meta} {ks_dir}",
    )
    parser.add_argument(
        "--no-spikeinterface-extract",
        action="store_false",
        dest="use_spikeinterface_extract",
        default=True,
        help="Disable spikeinterface fallback for waveform extraction. Default is to use spikeinterface if no template provided.",
    )
    parser.add_argument("--tprime-exe", type=Path, default=r"C:\Users\kouhi\TPrime-win\TPrime.exe", help="Path to TPrime executable")
    parser.add_argument("--tprime-syncperiod", type=float, default=1.0, help="TPrime -syncperiod value")
    parser.add_argument("--digital-lines", default=None, help="Comma list of digital lines to extract (default: tagging -> 0,1,2,4,7; else 0)")
    parser.add_argument("--stim-line", type=int, default=7, help="Digital line index used as stimulation")
    parser.add_argument("--stim-protocol-cleanup", action="store_true", help="Enable cleanup of debris stimulation pulses.")
    parser.add_argument("--stim-min-ibi", type=float, default=1.9, help="Minimum inter-pulse interval (s) for stimulation protocol start.")
    parser.add_argument("--stim-max-ibi", type=float, default=2.1, help="Maximum inter-pulse interval (s) for stimulation protocol start.")
    parser.add_argument("--stim-min-pulses", type=int, default=15, help="Minimum number of consecutive pulses to identify protocol start.")
    parser.add_argument("--mask-tol-ms", type=float, default=1.0, help="Mask spikes within +/- tol (ms) of stim edges")
    parser.add_argument(
        "--no-tagging",
        action="store_false",
        dest="tagging",
        help="Run in non-tagging mode (default is to run in tagging mode).",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter to use for templates")
    parser.add_argument("--rastermap-bin-ms", type=float, default=50.0, help="Bin size for Rastermap in milliseconds")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--catgt-gfix", default="0.40,0.10,0.02", help="CatGT global artifact correction parameters (e.g. '0.40,0.10,0.02')")
    parser.add_argument("--catgt-t-inline", action="store_true", default=False, help="Enable CatGT t_inline artifact correction")
    parser.add_argument("--ks-highpass-cutoff", type=float, default=400.0, help="Kilosort4 high-pass filter cutoff (Hz). Increase if you have high-frequency noise.")
    parser.add_argument("--ks-th-universal", type=float, default=8.0, help="Kilosort4 universal template threshold. Lower to find more units (default 9).")
    parser.add_argument("--ks-th-learned", type=float, default=7.0, help="Kilosort4 learned template threshold. Lower to find more units (default 8).")
    parser.set_defaults(tagging=True)
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    data_roots = []
    for item in args.data_root:
        for part in item.split(","):
            if part.strip():
                data_roots.append(Path(part.strip()))
    if args.sessions and len(args.sessions) == 1 and args.sessions[0].lower() == "all":
        args.sessions = None
    sessions = discover_sessions(data_roots, args.sessions, args.kilosort_output_subdir)
    digital_lines = (
        [int(x) for x in args.digital_lines.split(",")] if args.digital_lines else None
    )

    for session in sessions:
        if session["session_name"].endswith("_g1"):
            log(f"[{session['session_name']}] g1 folder; skipping all steps")
            continue
        log(f"=== Processing {session['session_name']} ===")
        if "catgt" in steps:
            run_catgt(session, args.catgt_template, args.catgt_dir, args.dry_run, args.catgt_gfix, args.catgt_t_inline)
        if "channelmap" in steps:
            run_channel_map(session, args.matlab, args.dry_run)
        if "kilosort" in steps:
            run_kilosort(
                session,
                args.kilosort_template,
                args.python_bin,
                args.dry_run,
                args.use_spikeinterface_kilosort,
                ks_highpass_cutoff=args.ks_highpass_cutoff,
                ks_th_universal=args.ks_th_universal,
                ks_th_learned=args.ks_th_learned,
            )
        if "digital" in steps:
            extract_digital_io(
                session,
                args.tagging,
                digital_lines,
                args.stim_line,
                args.dry_run,
                args.stim_protocol_cleanup,
                args.stim_min_ibi,
                args.stim_max_ibi,
                args.stim_min_pulses,
            )
        if "tprime" in steps:
            run_tprime(session, args.tprime_exe, args.tprime_syncperiod, args.dry_run)
        if "mask" in steps:
            build_spike_mask(session, args.stim_line, args.mask_tol_ms / 1000.0, args.dry_run)
        if "bombcell" in steps:
            run_bombcell(session, args.matlab, args.bombcell_path, args.dry_run)
        if "spikeinterface" in steps:
            run_spikeinterface_extract(
                session,
                args.spikeinterface_template,
                args.python_bin,
                args.dry_run,
                args.use_spikeinterface_extract,
            )
        if "waveform_metrics" in steps:
            export_waveforms_and_metrics(session, args.dry_run)
        if "classify" in steps:
            classify_cells(session, args.dry_run)

        if "lfp_csd" in steps:
            run_lfp_csd(session, args.dry_run)

        if "rastermap" in steps:
            subsets = get_rastermap_subsets(session)
            if not subsets:
                log(f"[{session['session_name']}] Could not determine subsets for rastermap (check BombCell/classification outputs)")
            else:
                for subset_name, unit_ids in subsets.items():
                    run_rastermap(
                        session, 
                        args.dry_run, 
                        subset_name=subset_name, 
                        target_unit_ids=unit_ids, 
                        bin_size_ms=args.rastermap_bin_ms
                    )


if __name__ == "__main__":
    main()
