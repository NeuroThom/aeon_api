"""Tests for the ONIX timing QC: HarpSync integrity and drift, clock sequence, readers."""

import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from dotmap import DotMap

import swc.aeon.qc.onix as _onix
from swc.aeon.io.api import chunk, to_datetime
from swc.aeon.qc.ephys import (
    harp_sync_drift,
    harp_sync_fits,
    harp_sync_integrity,
    onix_clock_files,
    onix_clock_sequence,
    onix_hub_offset,
)
from swc.aeon.qc.onix import HubSyncCounter, clock_reader
from swc.aeon.qc.reader import NOMINAL_CLOCK_HZ, HarpSync
from swc.aeon.qc.report import run_qc
from swc.aeon.schema.streams import Device

_PATCH = "swc.aeon.qc.ephys.load"
_ROOT = "/fake/root"
_START = pd.Timestamp("2025-06-04T16:00:00", tz="UTC")
_HARP0 = 3_831_899_129
_CLOCK0 = 404_205_298
_TICKS_PER_SECOND = 249_423_632  # measured on the AEONX1 social-ephys0.1 test recording
_PROBE_STEP = 8_333  # 250 MHz / 30 kHz
# Synthetic epochs (write_epoch) hold ten clock samples per second. A window edge placed half a
# sample before a whole second never lands on a sample. The in-window counts are then exact whatever
# the rounding of the HarpSync fit, which places an edge to within a few microseconds (_EDGE_TICKS).
_EDGE = 0.05
_EDGE_TICKS = 1_000


def harp_time(seconds: float) -> pd.Timestamp:
    """Harp seconds as a nanosecond Timestamp (the scalar ``to_datetime`` truncates to microseconds)."""
    return pd.Timestamp(to_datetime(pd.Index([seconds]))[0])


_SYNC_READER = HarpSync("NeuropixelsV2Beta_HarpSync_*")
_CLOCK_READER = clock_reader("NeuropixelsV2Beta_ProbeA_Clock_*")


def make_harp_sync(
    harp_times: list[int],
    clocks: list[int] | None = None,
    seconds_offset: int = 1,
) -> pd.DataFrame:
    """Build a HarpSync frame as ``load`` would return it (index is the ``Seconds`` column)."""
    if clocks is None:
        clocks = [_CLOCK0 + (h - harp_times[0]) * _TICKS_PER_SECOND for h in harp_times]
    seconds = [h + seconds_offset for h in harp_times]
    idx = pd.DatetimeIndex(to_datetime(pd.Index(seconds, dtype=float)), name="time")
    return pd.DataFrame(
        {"clock": clocks, "hub_clock": [c - 25 for c in clocks], "harp_time": harp_times},
        index=idx,
    )


# --- readers ---


def test_harp_sync_reader_reads_workflow_csv(tmp_path):
    """The HarpSync reader indexes on Seconds and names the three value columns."""
    path = tmp_path / "NeuropixelsV2Beta_HarpSync_2025-06-04T16-00-00.csv"
    path.write_text(
        "Seconds,Value.Clock,Value.HubClock,Value.HarpTime\n"
        "3831899130,404205298,404205273,3831899129\n"
        "3831899131,653628926,653628901,3831899130\n"
    )
    data = _SYNC_READER.read(path)
    assert list(data.columns) == ["clock", "hub_clock", "harp_time"]
    assert list(data.index) == [3831899130, 3831899131]
    assert data["harp_time"].iloc[0] == 3831899129


def test_clock_reader_is_uint64_binary_tagged_uniform(tmp_path):
    """clock_reader builds a swc.aeon Binary reader over uint64 ticks, tagged with uniform."""
    path = tmp_path / "NeuropixelsV2Beta_ProbeA_Clock_0.bin"
    np.array([1, 2, 2**40], dtype=np.uint64).tofile(path)
    data = _CLOCK_READER.read(path)
    assert list(data.columns) == ["clock"]
    assert list(data["clock"]) == [1, 2, 2**40]
    assert _CLOCK_READER.uniform is True
    assert clock_reader("X_Bno055_Clock_*", uniform=False).uniform is False


# --- harp_sync_integrity ---


def test_integrity_empty_data():
    """No data gives an empty frame with the documented schema and null attrs."""
    with patch(_PATCH, return_value=pd.DataFrame()):
        result = harp_sync_integrity(_ROOT, _SYNC_READER, _START)
    assert result.empty
    assert list(result.columns) == ["kind", "harp_step", "clock_step_ticks", "deviation_ticks", "device"]
    assert result.index.name == "time"
    assert result.attrs["data_found"] is False
    assert result.attrs["seconds_offset"] is None
    assert result.attrs["n_faults"] == 0


def test_integrity_keeps_every_step_with_offset_and_rate():
    """Every step is a row; a clean recording has no faults and the measured rate."""
    data = make_harp_sync([_HARP0 + i for i in range(6)], seconds_offset=1)
    with patch(_PATCH, return_value=data):
        result = harp_sync_integrity(_ROOT, _SYNC_READER, _START)
    assert len(result) == 5
    assert (result["kind"] == "ok").all()
    assert result.index[0] == data.index[1]
    assert result.attrs["n_faults"] == 0
    assert result.attrs["n_sync_events"] == 6
    assert result.attrs["seconds_offset"] == pytest.approx(1.0)
    assert result.attrs["clock_step_median_ticks"] == pytest.approx(_TICKS_PER_SECOND)
    assert result.attrs["clock_step_ppm"] == pytest.approx(-2305.47, abs=0.01)
    assert result.attrs["deviation_max_abs_ticks"] == pytest.approx(0.0)


def test_integrity_fixed_recording_has_zero_offset():
    """When the Seconds column equals harp_time the offset is zero."""
    data = make_harp_sync([_HARP0 + i for i in range(3)], seconds_offset=0)
    with patch(_PATCH, return_value=data):
        result = harp_sync_integrity(_ROOT, _SYNC_READER, _START)
    assert result.attrs["seconds_offset"] == pytest.approx(0.0)


def test_integrity_names_faults_and_measures_deviation():
    """Harp-time faults are named; a late pulse is measured, not classified."""
    # Clean run, then a 3 s gap, a repeat, a step back, recovery, and a 20 ms late pulse.
    harp = [_HARP0 + d for d in (0, 1, 2, 3, 4, 5, 8, 8, 7, 8, 9, 10)]
    clocks = [_CLOCK0 + (h - _HARP0) * _TICKS_PER_SECOND for h in harp]
    clocks[-1] += 5_000_000
    data = make_harp_sync(harp, clocks)
    with patch(_PATCH, return_value=data):
        result = harp_sync_integrity(_ROOT, _SYNC_READER, _START)
    faults = result[result["kind"] != "ok"]
    assert list(faults["kind"]) == ["harp_time_gap", "harp_time_duplicate", "harp_time_backwards"]
    gap = faults.iloc[0]
    assert gap["harp_step"] == 3
    assert gap["deviation_ticks"] == pytest.approx(0.0)
    assert faults.index[0] == data.index[6]
    assert np.isnan(faults.iloc[1]["deviation_ticks"])
    assert result.iloc[-1]["kind"] == "ok"
    assert result.iloc[-1]["deviation_ticks"] == pytest.approx(5_000_000)
    assert result.attrs["n_faults"] == 3
    assert result.attrs["n_harp_time_gap"] == 1
    assert result.attrs["deviation_max_abs_ticks"] == pytest.approx(5_000_000)
    assert (result["device"] == _SYNC_READER.pattern).all()


# --- harp_sync_drift ---


def test_drift_empty_data():
    """Fewer than two records gives an empty frame with null fit attrs."""
    with patch(_PATCH, return_value=make_harp_sync([_HARP0])):
        result = harp_sync_drift(_ROOT, _SYNC_READER, _START)
    assert result.empty
    assert list(result.columns) == ["residual_seconds", "chunk", "device"]
    assert result.index.name == "time"
    assert result.attrs["data_found"] is True
    assert result.attrs["fit"] is None


def test_drift_linear_recording():
    """A perfectly linear recording has near-zero residuals and the measured rate."""
    data = make_harp_sync([_HARP0 + i for i in range(10)])
    with patch(_PATCH, return_value=data):
        result = harp_sync_drift(_ROOT, _SYNC_READER, _START)
    assert len(result) == 10
    assert np.abs(result["residual_seconds"]).max() < 1e-6
    assert result.attrs["clock_rate_hz"] == pytest.approx(_TICKS_PER_SECOND, rel=1e-9)
    assert result.attrs["clock_rate_ppm"] == pytest.approx(-2305.47, abs=0.01)
    assert result.attrs["worst_chunk_max_abs_residual_ms"] < 1e-3
    assert "r2" not in result.attrs
    slope, intercept = result.attrs["fit"]
    assert slope * data["clock"].iloc[3] + intercept == pytest.approx(_HARP0 + 3)


def test_drift_fits_each_hourly_chunk_separately():
    """A rate change at the hour is absorbed by per-chunk fits; each chunk reports its rate."""
    t0 = 3_865_485_600  # a whole Harp hour
    harp = np.arange(t0 - 600, t0 + 600)
    clock = np.where(
        harp < t0,
        _CLOCK0 + (harp - harp[0]) * _TICKS_PER_SECOND,
        _CLOCK0 + (t0 - harp[0]) * _TICKS_PER_SECOND + (harp - t0) * (_TICKS_PER_SECOND + 2_500),
    )
    data = make_harp_sync(list(harp), list(clock), seconds_offset=0)
    with patch(_PATCH, return_value=data):
        result = harp_sync_drift(_ROOT, _SYNC_READER, _START)
    chunks = result.attrs["chunks"]
    assert result.attrs["n_chunks"] == 2
    assert list(chunks["n_sync_events"]) == [600, 600]
    assert chunks["clock_rate_ppm"].iloc[1] - chunks["clock_rate_ppm"].iloc[0] == pytest.approx(
        10.0, abs=0.1
    )
    assert result.attrs["worst_chunk_max_abs_residual_ms"] < 1e-3
    assert result["chunk"].nunique() == 2


def test_drift_splits_the_fit_at_a_clock_reset():
    """An ONIX restart resets the clock; each segment is fitted on its own and counted."""
    harp = [_HARP0 + i for i in range(20)]
    clock = [_CLOCK0 + i * _TICKS_PER_SECOND for i in range(10)]
    clock += [1_000 + i * (_TICKS_PER_SECOND + 2_500) for i in range(10)]
    data = make_harp_sync(harp, clock, seconds_offset=0)
    with patch(_PATCH, return_value=data):
        result = harp_sync_drift(_ROOT, _SYNC_READER, _START)
    chunks = result.attrs["chunks"]
    assert result.attrs["n_clock_resets"] == 1
    assert list(chunks["segment"]) == [0, 1]
    assert list(chunks["n_sync_events"]) == [10, 10]
    assert result.attrs["worst_chunk_max_abs_residual_ms"] < 1e-3
    assert chunks["clock_rate_ppm"].iloc[1] - chunks["clock_rate_ppm"].iloc[0] == pytest.approx(
        10.0, abs=0.1
    )


# --- onix_clock_files ---


def write_harp_sync(epoch_dir: Path, device: str, harp_times: list[int], clocks: list[int]) -> Path:
    """Write a HarpSync CSV as the workflow does, with Seconds one second late."""
    device_dir = epoch_dir / device
    device_dir.mkdir(parents=True, exist_ok=True)
    name = chunk(to_datetime(float(harp_times[0]))).strftime("%Y-%m-%dT%H-%M-%S")
    path = device_dir / f"{device}_HarpSync_{name}.csv"
    lines = ["Seconds,Value.Clock,Value.HubClock,Value.HarpTime"]
    lines += [f"{h + 1},{c},{c - 25},{h}" for h, c in zip(harp_times, clocks, strict=True)]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_epoch(epoch_dir: Path, seconds: int, per_second: int = 10) -> tuple[list[int], int]:
    """Write a HarpSync CSV and one ProbeA clock file per 20 s; return the ticks and sample step."""
    harp = [_HARP0 + i for i in range(seconds)]
    clocks = [_CLOCK0 + i * _TICKS_PER_SECOND for i in range(seconds)]
    write_harp_sync(epoch_dir, "NeuropixelsV2Beta", harp, clocks)
    step = _TICKS_PER_SECOND // per_second
    ticks = [_CLOCK0 + i * step for i in range(seconds * per_second)]
    size = 20 * per_second
    for index, offset in enumerate(range(0, len(ticks), size)):
        write_clock(epoch_dir, "NeuropixelsV2Beta", "ProbeA", index, ticks[offset : offset + size])
    return ticks, step


def write_clock(epoch_dir: Path, device: str, stream: str, index: int, ticks: list[int]) -> Path:
    """Write a numbered ONIX clock file in the Aeon layout."""
    device_dir = epoch_dir / device
    device_dir.mkdir(parents=True, exist_ok=True)
    path = device_dir / f"{device}_{stream}_Clock_{index}.bin"
    np.array(ticks, dtype=np.uint64).tofile(path)
    return path


def test_clock_files_without_harp_sync_fall_back_to_epoch_names(tmp_path):
    """Files sort numerically; without HarpSync, epochs are chosen by name with an earlier fallback."""
    e1 = tmp_path / "2025-06-04T15-00-00"
    e2 = tmp_path / "2025-06-04T16-00-00"
    e3 = tmp_path / "2025-06-04T18-00-00"
    for i in (0, 2, 10, 1):
        write_clock(e2, "NeuropixelsV2Beta", "ProbeA", i, [i])
    write_clock(e1, "NeuropixelsV2Beta", "ProbeA", 0, [0])
    write_clock(e3, "NeuropixelsV2Beta", "ProbeA", 0, [0])
    write_clock(e2, "NeuropixelsV2Beta", "ProbeB", 0, [0])  # other stream, ignored

    files = onix_clock_files(tmp_path, _CLOCK_READER, _START, end=_START + pd.Timedelta(hours=1))
    assert [f.path.parent.parent.name for f in files] == ["2025-06-04T16-00-00"] * 4
    assert [f.path.name.rsplit("_", 1)[-1] for f in files] == ["0.bin", "1.bin", "2.bin", "10.bin"]
    assert all(f.fit is None and f.tick_lo == -np.inf for f in files)

    # A window starting mid-epoch falls back to the latest epoch that started before it.
    files = onix_clock_files(tmp_path, _CLOCK_READER, _START + pd.Timedelta(minutes=30))
    epochs = [f.path.parent.parent.name for f in files]
    assert epochs == ["2025-06-04T16-00-00"] * 4 + ["2025-06-04T18-00-00"]

    # A root that is itself an epoch directory is used as is; a list of roots is walked in order.
    files = onix_clock_files([e3, e1], _CLOCK_READER, _START)
    assert [f.path.parent.parent.name for f in files] == ["2025-06-04T18-00-00", "2025-06-04T15-00-00"]


def test_clock_files_selected_by_tick_overlap_with_the_window(tmp_path):
    """With HarpSync, the window becomes tick bounds and only files crossing them are kept."""
    epoch = tmp_path / "2025-06-04T16-25-29"
    ticks, step = write_epoch(epoch, seconds=60)
    start = to_datetime(float(_HARP0) + 25 - _EDGE)
    end = to_datetime(float(_HARP0) + 45 - _EDGE)

    files = onix_clock_files(tmp_path, _CLOCK_READER, start, end=end)
    assert [f.path.name.rsplit("_", 1)[-1] for f in files] == ["1.bin", "2.bin"]
    assert files[0].first_tick == ticks[200]
    assert files[0].tick_lo == pytest.approx(ticks[250] - step / 2, abs=_EDGE_TICKS)
    assert files[0].tick_hi == pytest.approx(ticks[450] - step / 2, abs=_EDGE_TICKS)
    assert files[0].rate_hz == pytest.approx(_TICKS_PER_SECOND)

    # A window that started before the recording, with no end, keeps every file.
    files = onix_clock_files(tmp_path, _CLOCK_READER, to_datetime(float(_HARP0 - 3600)))
    assert len(files) == 3
    assert files[-1].tick_hi == np.inf

    # A window after the recording keeps nothing.
    assert onix_clock_files(tmp_path, _CLOCK_READER, to_datetime(float(_HARP0 + 100))) == []


def test_clock_files_match_domains_to_fits_after_a_restart(tmp_path):
    """Files after a clock reset are placed with the fit of their own clock domain."""
    epoch = tmp_path / "2025-06-04T16-25-29"
    harp = [_HARP0 + i for i in range(20)]
    clock = [_CLOCK0 + i * _TICKS_PER_SECOND for i in range(10)]
    clock += [1_000 + i * _TICKS_PER_SECOND for i in range(10)]
    write_harp_sync(epoch, "NeuropixelsV2Beta", harp, clock)
    step = _TICKS_PER_SECOND // 10
    write_clock(epoch, "NeuropixelsV2Beta", "ProbeA", 0, [_CLOCK0 + i * step for i in range(100)])
    write_clock(epoch, "NeuropixelsV2Beta", "ProbeA", 1, [1_000 + i * step for i in range(100)])
    assert len(harp_sync_fits(epoch, "NeuropixelsV2Beta")) == 2

    start = to_datetime(float(_HARP0) + 5 - _EDGE)
    end = to_datetime(float(_HARP0) + 15 - _EDGE)
    files = onix_clock_files(tmp_path, _CLOCK_READER, start, end=end)
    assert [f.path.name.rsplit("_", 1)[-1] for f in files] == ["0.bin", "1.bin"]
    assert files[0].tick_lo == pytest.approx(_CLOCK0 + 49.5 * step, abs=_EDGE_TICKS)
    assert files[1].tick_hi == pytest.approx(1_000 + 49.5 * step, abs=_EDGE_TICKS)

    result = onix_clock_sequence(tmp_path, _CLOCK_READER, start, end=end)
    assert result.attrs["n_samples"] == 100
    # The carried sample across the reset is one backwards step, placed on Harp time.
    assert list(result["kind"]) == ["backwards"]
    assert abs(result.index[0] - harp_time(_HARP0 + 10)) < pd.Timedelta(microseconds=1)


# --- onix_clock_sequence ---


def test_clock_sequence_no_files():
    """No files gives an empty frame with zero samples."""
    result = onix_clock_sequence("/nonexistent/root", _CLOCK_READER, _START)
    assert result.empty
    assert list(result.columns) == [
        "kind",
        "clock_ticks",
        "step_ticks",
        "step_seconds",
        "file",
        "index_in_file",
        "device",
    ]
    assert result.attrs["data_found"] is False
    assert result.attrs["n_files"] == 0


def test_clock_sequence_clean_two_files_and_blocks(tmp_path):
    """A steady step across file and block boundaries produces no rows."""
    epoch = tmp_path / "2025-06-04T16-00-00"
    ticks = [1_000 + i * _PROBE_STEP for i in range(50)]
    write_clock(epoch, "NeuropixelsV2Beta", "ProbeA", 0, ticks[:30])
    write_clock(epoch, "NeuropixelsV2Beta", "ProbeA", 1, ticks[30:])
    result = onix_clock_sequence(tmp_path, _CLOCK_READER, _START, block_samples=7)
    assert result.empty
    assert result.attrs["data_found"] is True
    assert result.attrs["n_samples"] == 50
    assert result.attrs["n_files"] == 2
    assert result.attrs["nominal_step_ticks"] == _PROBE_STEP
    assert result.attrs["inferred_rate_hz"] == pytest.approx(NOMINAL_CLOCK_HZ / _PROBE_STEP)


def test_clock_sequence_flags_every_kind(tmp_path):
    """Backwards, duplicate and jump are each named, with file positions."""
    epoch = tmp_path / "2025-06-04T16-00-00"
    base = [1_000 + i * _PROBE_STEP for i in range(40)]
    ticks = list(base)
    ticks[10] = base[10] + 2**32  # single-sample glitch: up by 2^32, back down next sample
    ticks[20] = base[19]  # duplicate of the previous sample, then a double step to recover
    ticks[25] = base[24] - 100  # backwards, then a double step (plus 100) to recover
    ticks[30] = base[30] + 10 * _PROBE_STEP  # jump forward, then 9 steps back onto the sequence
    write_clock(epoch, "NeuropixelsV2Beta", "ProbeA", 0, ticks[:15])
    write_clock(epoch, "NeuropixelsV2Beta", "ProbeA", 1, ticks[15:])

    result = onix_clock_sequence(tmp_path, _CLOCK_READER, _START, block_samples=8)

    kinds = list(result["kind"])
    assert kinds == [
        "jump",
        "backwards",
        "duplicate",
        "jump",
        "backwards",
        "jump",
        "jump",
        "backwards",
    ]
    assert result.iloc[0]["step_ticks"] == 2**32 + _PROBE_STEP
    assert result.iloc[1]["step_ticks"] == -(2**32) + _PROBE_STEP
    assert result.iloc[0]["index_in_file"] == 10
    assert result.iloc[0]["file"].endswith("_Clock_0.bin")
    dup = result.iloc[2]
    assert dup["file"].endswith("_Clock_1.bin")
    assert dup["index_in_file"] == 5
    assert dup["step_seconds"] == 0.0
    assert list(result["index_in_file"]) == [10, 11, 5, 6, 10, 11, 15, 16]
    assert result.attrs["n_duplicate"] == 1
    assert result.attrs["n_backwards"] == 3
    assert result.attrs["n_jump"] == 4
    assert result.attrs["n_samples"] == 40
    assert result.index.isna().all()


def test_clock_sequence_checks_only_the_window_and_uses_harp_time(tmp_path):
    """With HarpSync, only in-window samples are checked; rows carry Harp time and the fitted rate."""
    epoch = tmp_path / "2025-06-04T16-25-29"
    ticks, step = write_epoch(epoch, seconds=60)
    paths = sorted(epoch.glob("*/*_Clock_*.bin"))
    data = np.fromfile(paths[1], dtype=np.uint64)
    data[100] = data[99]  # duplicate at second 30, inside the window
    data.tofile(paths[1])
    data = np.fromfile(paths[0], dtype=np.uint64)
    data[50] = data[49]  # duplicate at second 5, outside the window
    data.tofile(paths[0])

    start = to_datetime(float(_HARP0) + 25 - _EDGE)
    end = to_datetime(float(_HARP0) + 45 - _EDGE)
    result = onix_clock_sequence(tmp_path, _CLOCK_READER, start, end=end, block_samples=64)
    assert result.attrs["n_samples"] == 200
    assert result.attrs["n_files"] == 2
    assert list(result.attrs["files"]["n_samples"]) == [150, 50]
    assert result.attrs["files"]["first_tick"].iloc[0] == ticks[250]
    assert list(result["kind"]) == ["duplicate", "jump"]
    assert list(result["index_in_file"]) == [100, 101]
    expected = harp_time(_HARP0 + 299 * step / _TICKS_PER_SECOND)
    assert abs(result.index[0] - expected) < pd.Timedelta(microseconds=1)
    assert result.index.tz == datetime.UTC
    assert result.attrs["clock_rate_hz"] == pytest.approx(_TICKS_PER_SECOND)
    assert result.iloc[1]["step_seconds"] == pytest.approx(2 * step / _TICKS_PER_SECOND)


def test_clock_sequence_non_uniform_stream_ignores_irregular_steps(tmp_path):
    """A non-uniform stream reports backwards steps and duplicates but not jumps."""
    epoch = tmp_path / "2025-06-04T16-25-34"
    ticks = [1_000, 2_000, 5_000, 5_500, 5_400, 20_000, 20_000 + 2**32, 21_000, 21_000]
    write_clock(epoch, "NeuropixelsV2Beta", "Bno055", 0, ticks)
    reader = clock_reader("NeuropixelsV2Beta_Bno055_Clock_*", uniform=False)
    result = onix_clock_sequence(tmp_path, reader, _START)
    assert list(result["kind"]) == ["backwards", "backwards", "duplicate"]
    assert result.attrs["n_jump"] == 0


# --- onix_hub_offset ---


def write_binary(epoch_dir: Path, device: str, name: str, index: int, ticks: list[int]) -> None:
    """Write a numbered uint64 ONIX binary file in the Aeon layout."""
    device_dir = epoch_dir / device
    device_dir.mkdir(parents=True, exist_ok=True)
    np.array(ticks, dtype=np.uint64).tofile(device_dir / f"{device}_{name}_{index}.bin")


_HUB_READER = HubSyncCounter("NeuropixelsV2Beta_ProbeA_HubSyncCounter_*").reader


def test_hub_offset_summarises_every_file(tmp_path):
    """One row per file with the offset range; the distribution and nominal are kept."""
    epoch = tmp_path / "2025-06-04T16-00-00"
    clock = [1_000_000 + i * _PROBE_STEP for i in range(40)]
    hub = [c - 8_331 + (i % 3) for i, c in enumerate(clock)]
    for index, part in ((0, slice(0, 25)), (1, slice(25, None))):
        write_binary(epoch, "NeuropixelsV2Beta", "ProbeA_Clock", index, clock[part])
        write_binary(epoch, "NeuropixelsV2Beta", "ProbeA_HubSyncCounter", index, hub[part])
    result = onix_hub_offset(tmp_path, _CLOCK_READER, _HUB_READER, _START, block_samples=7)
    assert list(result.columns) == [
        "file",
        "n_samples",
        "min_offset_ticks",
        "mean_offset_ticks",
        "max_offset_ticks",
        "length_mismatch",
        "device",
    ]
    assert list(result["n_samples"]) == [25, 15]
    assert list(result["min_offset_ticks"]) == [8_329, 8_329]
    assert list(result["max_offset_ticks"]) == [8_331, 8_331]
    assert result.index.isna().all()
    assert result.attrs["n_samples"] == 40
    assert result.attrs["nominal_offset_ticks"] == 8_330
    assert result.attrs["deviation_histogram"].sum() == 40
    assert result.attrs["deviation_max_abs_ticks"] == 1
    assert not result["length_mismatch"].any()


def test_hub_offset_keeps_largest_deviations_with_location(tmp_path):
    """A relock is not classified, it is measured: the worst samples keep file and index."""
    epoch = tmp_path / "2025-06-04T16-00-00"
    clock = [1_000_000 + i * _PROBE_STEP for i in range(30)]
    hub = [c - 8_331 for c in clock]
    for i in range(5, 12):
        hub[i] -= 5_000
    hub[20] += 1_000
    write_binary(epoch, "NeuropixelsV2Beta", "ProbeA_Clock", 0, clock)
    write_binary(epoch, "NeuropixelsV2Beta", "ProbeA_HubSyncCounter", 0, hub)
    result = onix_hub_offset(tmp_path, _CLOCK_READER, _HUB_READER, _START, block_samples=8)
    worst = result.attrs["worst"]
    assert result.attrs["deviation_max_abs_ticks"] == 5_000
    assert [w["deviation_ticks"] for w in worst[:7]] == [5_000] * 7
    assert sorted(w["index_in_file"] for w in worst[:7]) == list(range(5, 12))
    assert worst[7]["deviation_ticks"] == -1_000
    assert worst[7]["index_in_file"] == 20
    assert worst[0]["time"] is None
    assert result.iloc[0]["max_offset_ticks"] == 8_331 + 5_000


def test_hub_offset_flags_length_mismatch(tmp_path):
    """Clock and hub files with different sample counts are marked on the file row."""
    epoch = tmp_path / "2025-06-04T16-00-00"
    clock = [1_000_000 + i * _PROBE_STEP for i in range(10)]
    write_binary(epoch, "NeuropixelsV2Beta", "ProbeA_Clock", 0, clock)
    write_binary(epoch, "NeuropixelsV2Beta", "ProbeA_HubSyncCounter", 0, [c - 8_331 for c in clock[:9]])
    result = onix_hub_offset(tmp_path, _CLOCK_READER, _HUB_READER, _START)
    assert bool(result.iloc[0]["length_mismatch"]) is True
    assert result.attrs["n_samples"] == 9


def write_hub_files(epoch_dir: Path, hub: list[int], size: int = 200) -> None:
    """Write ProbeA hub clock files in the same 20 s pieces as ``write_epoch``."""
    for index, offset in enumerate(range(0, len(hub), size)):
        write_binary(
            epoch_dir, "NeuropixelsV2Beta", "ProbeA_HubSyncCounter", index, hub[offset : offset + size]
        )


def test_hub_offset_uses_only_the_window_and_carries_harp_time(tmp_path):
    """With HarpSync, only in-window samples count; file rows and worst samples carry Harp time."""
    epoch = tmp_path / "2025-06-04T16-25-29"
    ticks, step = write_epoch(epoch, seconds=60)
    hub = [t - 8_331 for t in ticks]
    hub[100] -= 5_000  # second 10, outside the window
    hub[300] -= 3_000  # second 30, inside the window
    write_hub_files(epoch, hub)

    start = to_datetime(float(_HARP0) + 25 - _EDGE)
    end = to_datetime(float(_HARP0) + 45 - _EDGE)
    result = onix_hub_offset(tmp_path, _CLOCK_READER, _HUB_READER, start, end=end, block_samples=64)
    assert list(result["n_samples"]) == [150, 50]
    assert result.attrs["deviation_max_abs_ticks"] == 3_000
    assert result.attrs["worst"][0]["index_in_file"] == 100
    assert abs(result.index[0] - harp_time(_HARP0 + 25)) < pd.Timedelta(microseconds=1)
    expected = harp_time(_HARP0 + 300 * step / _TICKS_PER_SECOND)
    assert abs(result.attrs["worst"][0]["time"] - expected) < pd.Timedelta(microseconds=1)


# --- run_qc over a behaviour root and an ephys root ---


def test_run_qc_finds_ephys_for_a_behaviour_window_in_a_second_root(tmp_path):
    """The ephys root is searched by time overlap, whichever recording started first."""
    behaviour = tmp_path / "AEON3" / "abcBehav01"
    ephys = tmp_path / "AEON3" / "abcEphys01"
    (behaviour / "2025-06-04T16-26-00").mkdir(parents=True)
    epoch = ephys / "2025-06-04T16-25-29"
    ticks, _ = write_epoch(epoch, seconds=60)
    write_hub_files(epoch, [t - 8_331 for t in ticks])
    schema = DotMap([Device("NeuropixelsV2Beta", _onix.NeuropixelsV2Beta, _onix.ProbeA)])

    start = to_datetime(float(_HARP0) + 31 - _EDGE)
    end = to_datetime(float(_HARP0) + 45 - _EDGE)
    results = run_qc([behaviour, ephys], schema, start, end=end)

    assert results["NeuropixelsV2Beta.HarpSync"].attrs["n_sync_events"] == 14
    assert results["NeuropixelsV2Beta.HarpSync.drift"].attrs["fit"] is not None
    clock = results["NeuropixelsV2Beta.ProbeAClock"]
    assert clock.attrs["n_samples"] == 140
    assert list(clock.attrs["files"]["file"].str[-5:]) == ["1.bin", "2.bin"]
    assert clock.empty
    hub_result = results["NeuropixelsV2Beta.ProbeAHubSyncCounter"]
    assert hub_result.attrs["n_samples"] == 140
    assert hub_result.attrs["deviation_max_abs_ticks"] == 0
