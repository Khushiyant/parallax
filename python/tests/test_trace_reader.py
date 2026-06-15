"""Tests for prlx.trace_reader."""

import struct
import tempfile
from pathlib import Path

import pytest

from prlx.trace_reader import (
    EVENT_NAMES,
    HEADER_SIZE,
    PRLX_FLAG_COMPRESS,
    PRLX_FLAG_HISTORY,
    PRLX_FLAG_SNAPSHOT,
    PRLX_MAGIC,
    PRLX_VERSION,
    TRACE_EVENT_SIZE,
    WARP_BUFFER_HEADER_SIZE,
    TraceEvent,
    read_trace,
)

from .conftest import (
    DEFAULT_EVENTS_PER_WARP,
    _EVENT_FMT,
    _HEADER_FMT,
    _WARP_HEADER_FMT,
    make_event,
    make_trace,
)


# ============================================================
# Header parsing
# ============================================================


class TestHeaderParsing:
    def test_read_valid_trace(self, trace_builder):
        path = trace_builder(
            kernel_name="my_test_kernel",
            kernel_hash=0xDEADBEEF,
            grid_dim=(4, 2, 1),
            block_dim=(128, 1, 1),
            cuda_arch=86,
            timestamp=9999999,
            sample_rate=4,
            warp_events=[
                [(0x1000, 0, 1, 0xFFFFFFFF, 42)],
            ],
        )
        trace = read_trace(str(path))
        h = trace.header

        assert h.magic == PRLX_MAGIC
        assert h.version == PRLX_VERSION
        assert h.kernel_name == "my_test_kernel"
        assert h.kernel_name_hash == 0xDEADBEEF
        assert h.grid_dim == (4, 2, 1)
        assert h.block_dim == (128, 1, 1)
        assert h.cuda_arch == 86
        assert h.timestamp == 9999999
        assert h.sample_rate == 4
        assert h.total_warp_slots == 1

    def test_invalid_magic_rejected(self, tmp_path):
        bad_file = tmp_path / "bad.prlx"
        # Write enough bytes but with wrong magic
        data = struct.pack("<Q", 0xBADBADBADBADBAD0) + b"\x00" * (HEADER_SIZE - 8)
        bad_file.write_bytes(data)

        with pytest.raises(ValueError, match="Invalid magic"):
            read_trace(str(bad_file))

    def test_invalid_version_rejected(self, tmp_path):
        bad_file = tmp_path / "bad_ver.prlx"
        # Correct magic but wrong version
        data = struct.pack("<QI", PRLX_MAGIC, 99) + b"\x00" * (HEADER_SIZE - 12)
        bad_file.write_bytes(data)

        with pytest.raises(ValueError, match="Unsupported version"):
            read_trace(str(bad_file))

    def test_file_too_small(self, tmp_path):
        tiny_file = tmp_path / "tiny.prlx"
        tiny_file.write_bytes(b"\x00" * 32)

        with pytest.raises(ValueError, match="too small"):
            read_trace(str(tiny_file))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_trace("/nonexistent/path/trace.prlx")

    def test_kernel_name_decoding(self, trace_builder):
        # Max-length name (63 chars + null)
        long_name = "a" * 63
        path = trace_builder(kernel_name=long_name)
        trace = read_trace(str(path))
        assert trace.header.kernel_name == long_name

    def test_kernel_name_with_null(self, trace_builder):
        path = trace_builder(kernel_name="short")
        trace = read_trace(str(path))
        assert trace.header.kernel_name == "short"


# ============================================================
# Event parsing
# ============================================================


class TestEventParsing:
    def test_events_parsed_correctly(self, trace_builder):
        path = trace_builder(
            warp_events=[
                [(0xABCD, 0, 1, 0xFFFF0000, 12345)],
            ],
        )
        trace = read_trace(str(path))
        w = trace.warp(0)
        assert w.num_events == 1
        evt = w.events[0]
        assert evt.site_id == 0xABCD
        assert evt.event_type == 0
        assert evt.branch_dir == 1
        assert evt.active_mask == 0xFFFF0000
        assert evt.value_a == 12345

    def test_multiple_warps(self, trace_builder):
        path = trace_builder(
            warp_events=[
                [(0x1000, 0, 1, 0xFFFFFFFF, 1), (0x2000, 0, 0, 0xFFFFFFFF, 2)],
                [(0x3000, 0, 1, 0xFFFFFFFF, 3)],
                [],
            ],
        )
        trace = read_trace(str(path))
        assert trace.num_warps == 3
        assert trace.warp(0).num_events == 2
        assert trace.warp(1).num_events == 1
        assert trace.warp(2).num_events == 0

    def test_zero_events_warp(self, trace_builder):
        path = trace_builder(warp_events=[[]])
        trace = read_trace(str(path))
        assert trace.warp(0).num_events == 0
        assert trace.warp(0).events == []

    def test_event_type_names(self):
        assert EVENT_NAMES[0] == "Branch"
        assert EVENT_NAMES[1] == "SharedMemStore"
        assert EVENT_NAMES[2] == "Atomic"
        assert EVENT_NAMES[3] == "FuncEntry"
        assert EVENT_NAMES[4] == "FuncExit"
        assert EVENT_NAMES[5] == "Switch"
        assert EVENT_NAMES[6] == "GlobalStore"
        assert len(EVENT_NAMES) == 7


# ============================================================
# TraceEvent properties
# ============================================================


class TestTraceEventProperties:
    def test_is_branch(self):
        evt = TraceEvent(site_id=0, event_type=0, branch_dir=0, active_mask=0, value_a=0)
        assert evt.is_branch is True

        evt2 = TraceEvent(site_id=0, event_type=1, branch_dir=0, active_mask=0, value_a=0)
        assert evt2.is_branch is False

    def test_branch_taken(self):
        taken = TraceEvent(site_id=0, event_type=0, branch_dir=1, active_mask=0, value_a=0)
        assert taken.branch_taken is True

        not_taken = TraceEvent(site_id=0, event_type=0, branch_dir=0, active_mask=0, value_a=0)
        assert not_taken.branch_taken is False

    def test_active_thread_count(self):
        evt_all = TraceEvent(site_id=0, event_type=0, branch_dir=0, active_mask=0xFFFFFFFF, value_a=0)
        assert evt_all.active_thread_count == 32

        evt_half = TraceEvent(site_id=0, event_type=0, branch_dir=0, active_mask=0x0000FFFF, value_a=0)
        assert evt_half.active_thread_count == 16

        evt_none = TraceEvent(site_id=0, event_type=0, branch_dir=0, active_mask=0, value_a=0)
        assert evt_none.active_thread_count == 0

    def test_event_type_name(self):
        evt = TraceEvent(site_id=0, event_type=0, branch_dir=0, active_mask=0, value_a=0)
        assert evt.event_type_name == "Branch"

        evt2 = TraceEvent(site_id=0, event_type=99, branch_dir=0, active_mask=0, value_a=0)
        assert evt2.event_type_name == "Unknown(99)"


# ============================================================
# TraceData API
# ============================================================


class TestTraceDataAPI:
    def test_total_events(self, trace_builder):
        path = trace_builder(
            warp_events=[
                [(0x1000, 0, 1, 0xFFFFFFFF, 1), (0x2000, 0, 0, 0xFFFFFFFF, 2)],
                [(0x3000, 0, 1, 0xFFFFFFFF, 3)],
                [],
            ],
        )
        trace = read_trace(str(path))
        assert trace.total_events == 3

    def test_total_overflows(self, trace_builder):
        path = trace_builder(
            warp_events=[
                [(0x1000, 0, 1, 0xFFFFFFFF, 1)],
                [(0x2000, 0, 0, 0xFFFFFFFF, 2)],
            ],
            warp_overflows=[5, 10],
        )
        trace = read_trace(str(path))
        assert trace.total_overflows == 15

    def test_divergent_warps(self, trace_builder):
        path_a = trace_builder(
            warp_events=[
                [(0x1000, 0, 1, 0xFFFFFFFF, 1), (0x2000, 0, 0, 0xFFFFFFFF, 2)],
                [(0x3000, 0, 1, 0xFFFFFFFF, 3)],
            ],
        )
        path_b = trace_builder(
            warp_events=[
                [(0x1000, 0, 1, 0xFFFFFFFF, 1)],  # 1 event vs 2 in A
                [(0x3000, 0, 1, 0xFFFFFFFF, 3)],   # same count
            ],
        )
        trace_a = read_trace(str(path_a))
        trace_b = read_trace(str(path_b))
        divergent = trace_a.divergent_warps(trace_b)
        assert divergent == [0]  # Only warp 0 has different event count

    def test_warp_index_bounds(self, trace_builder):
        path = trace_builder(warp_events=[[]])
        trace = read_trace(str(path))
        with pytest.raises(IndexError):
            trace.warp(1)
        with pytest.raises(IndexError):
            trace.warp(-1)


# ============================================================
# Header properties
# ============================================================


class TestHeaderProperties:
    def test_has_history(self, trace_builder):
        path = trace_builder(
            history_depth=8,
            history_data=[[(0x1000, 42, 0)]],
            warp_events=[[(0x1000, 0, 1, 0xFFFFFFFF, 42)]],
        )
        trace = read_trace(str(path))
        assert trace.header.has_history is True

    def test_no_history_when_depth_zero(self, trace_builder):
        path = trace_builder()
        trace = read_trace(str(path))
        assert trace.header.has_history is False

    def test_has_snapshot(self, trace_builder):
        snap = {
            "site_id": 0x1000,
            "active_mask": 0xFFFFFFFF,
            "seq": 0,
            "cmp_predicate": 38,
            "lhs_values": list(range(32)),
            "rhs_values": [10] * 32,
        }
        path = trace_builder(
            snapshot_depth=4,
            snapshot_data=[[snap]],
            warp_events=[[(0x1000, 0, 1, 0xFFFFFFFF, 0)]],
        )
        trace = read_trace(str(path))
        assert trace.header.has_snapshot is True

    def test_total_blocks(self, trace_builder):
        path = trace_builder(grid_dim=(4, 2, 3))
        trace = read_trace(str(path))
        assert trace.header.total_blocks == 24

    def test_threads_per_block(self, trace_builder):
        path = trace_builder(block_dim=(128, 2, 1))
        trace = read_trace(str(path))
        assert trace.header.threads_per_block == 256

    def test_is_sampled(self, trace_builder):
        path = trace_builder(sample_rate=4, flags=PRLX_FLAG_SNAPSHOT)  # using a flag to test sampled
        trace = read_trace(str(path))
        # sample_rate=4 but PRLX_FLAG_SAMPLED not set → is_sampled should be False
        assert trace.header.is_sampled is False


# ============================================================
# History & Snapshot sections
# ============================================================


class TestHistorySection:
    def test_history_entries_parsed(self, trace_builder):
        path = trace_builder(
            warp_events=[[(0x1000, 0, 1, 0xFFFFFFFF, 42)]],
            history_depth=8,
            history_data=[
                [(0x1000, 100, 0), (0x1000, 200, 1), (0x2000, 300, 2)],
            ],
        )
        trace = read_trace(str(path))
        w = trace.warp(0)
        assert len(w.history) == 3
        # Should be sorted by seq
        assert w.history[0].seq == 0
        assert w.history[0].value == 100
        assert w.history[2].seq == 2
        assert w.history[2].value == 300

    def test_no_history_when_flag_unset(self, trace_builder):
        path = trace_builder(warp_events=[[]])
        trace = read_trace(str(path))
        assert trace.warp(0).history == []


class TestSnapshotSection:
    def test_snapshot_entries_parsed(self, trace_builder):
        snap = {
            "site_id": 0xABCD,
            "active_mask": 0xFFFFFFFF,
            "seq": 0,
            "cmp_predicate": 38,
            "lhs_values": list(range(32)),
            "rhs_values": [10] * 32,
        }
        path = trace_builder(
            warp_events=[[(0xABCD, 0, 1, 0xFFFFFFFF, 0)]],
            snapshot_depth=4,
            snapshot_data=[[snap]],
        )
        trace = read_trace(str(path))
        w = trace.warp(0)
        assert len(w.snapshots) == 1
        s = w.snapshots[0]
        assert s.site_id == 0xABCD
        assert s.cmp_predicate == 38
        assert s.lhs_values[5] == 5
        assert s.rhs_values[0] == 10
        assert len(s.lhs_values) == 32
        assert len(s.rhs_values) == 32


# ============================================================
# Cross-language format verification
# ============================================================


class TestBinaryFormat:
    def test_struct_sizes(self):
        assert struct.calcsize(_HEADER_FMT) == 160
        assert struct.calcsize(_EVENT_FMT) == 16
        assert struct.calcsize(_WARP_HEADER_FMT) == 16

    def test_header_field_offsets(self, trace_builder):
        path = trace_builder(
            kernel_name="test_kernel",
            kernel_hash=0x1234567890ABCDEF,
            grid_dim=(1, 1, 1),
            block_dim=(32, 1, 1),
            timestamp=0xDEADCAFEBABE0000,
            cuda_arch=80,
            events_per_warp=128,
        )
        raw = path.read_bytes()

        assert struct.unpack_from("<Q", raw, 0)[0] == PRLX_MAGIC          # offset 0: magic
        assert struct.unpack_from("<I", raw, 8)[0] == PRLX_VERSION        # offset 8: version
        assert struct.unpack_from("<Q", raw, 16)[0] == 0x1234567890ABCDEF # offset 16: hash
        assert raw[24:35] == b"test_kernel"                                # offset 24: name
        assert raw[35] == 0
        assert struct.unpack_from("<I", raw, 120)[0] == 128               # offset 120: events_per_warp
        assert raw[124:128] == b"\x00\x00\x00\x00"                        # offset 124: padding
        assert struct.unpack_from("<Q", raw, 128)[0] == 0xDEADCAFEBABE0000 # offset 128: timestamp
        assert struct.unpack_from("<I", raw, 136)[0] == 80                 # offset 136: cuda_arch

    def test_event_field_offsets(self):
        raw = make_event(site_id=0xABCD1234, event_type=2, branch_dir=1,
                         active_mask=0xFFFF0000, value_a=0xDEADBEEF)
        assert len(raw) == 16
        assert struct.unpack_from("<I", raw, 0)[0] == 0xABCD1234
        assert raw[4] == 2   # event_type
        assert raw[5] == 1   # branch_dir
        assert raw[6:8] == b"\x00\x00"  # _reserved
        assert struct.unpack_from("<I", raw, 8)[0] == 0xFFFF0000
        assert struct.unpack_from("<I", raw, 12)[0] == 0xDEADBEEF


# ============================================================
# Ring buffer overflow tests
# ============================================================


class TestOverflow:
    def test_overflow_count_preserved(self, trace_builder):
        path = trace_builder(
            warp_events=[[(0x1000, 0, 1, 0xFFFFFFFF, 42)]],
            warp_overflows=[500],
        )
        trace = read_trace(str(path))
        assert trace.warp(0).overflow_count == 500
        assert trace.warp(0).num_events == 1
        assert trace.total_overflows == 500

    def test_overflow_multiple_warps(self, trace_builder):
        path = trace_builder(
            warp_events=[
                [(0x1000, 0, 1, 0xFFFFFFFF, 1)],
                [(0x2000, 0, 0, 0xFFFFFFFF, 2)],
                [],
            ],
            warp_overflows=[100, 200, 50],
        )
        trace = read_trace(str(path))
        assert trace.warp(0).overflow_count == 100
        assert trace.warp(1).overflow_count == 200
        assert trace.warp(2).overflow_count == 50
        assert trace.total_overflows == 350

    def test_full_buffer_with_overflow(self, trace_builder):
        full_events = [(i, 0, 1, 0xFFFFFFFF, i) for i in range(DEFAULT_EVENTS_PER_WARP)]
        path = trace_builder(
            warp_events=[full_events],
            warp_overflows=[1000],
        )
        trace = read_trace(str(path))
        w = trace.warp(0)
        assert w.num_events == DEFAULT_EVENTS_PER_WARP
        assert w.overflow_count == 1000
        assert len(w.events) == DEFAULT_EVENTS_PER_WARP
        assert w.events[0].site_id == 0
        assert w.events[DEFAULT_EVENTS_PER_WARP - 1].site_id == DEFAULT_EVENTS_PER_WARP - 1

    def test_num_events_capped_at_events_per_warp(self, tmp_path):
        events_per_warp = 4
        name_padded = b"test_kernel" + b"\x00" * 53
        header = struct.pack(
            _HEADER_FMT,
            PRLX_MAGIC, PRLX_VERSION, 0,
            0x1234567890ABCDEF,
            name_padded,
            1, 1, 1,
            32, 1, 1,
            1, 1,
            events_per_warp,
            1234567890, 80,
            0, 0,
            1,
            0, 0,
        )
        warp_hdr = struct.pack(_WARP_HEADER_FMT, 10, 6, 10, 0)
        events_data = b""
        for i in range(events_per_warp):
            events_data += make_event(site_id=i, value_a=i)

        trace_file = tmp_path / "overflow.prlx"
        trace_file.write_bytes(header + warp_hdr + events_data)

        trace = read_trace(str(trace_file))
        w = trace.warp(0)
        assert w.num_events == events_per_warp
        assert len(w.events) == events_per_warp
        assert w.overflow_count == 6


# ============================================================
# Zstd compression tests
# ============================================================


class TestCompression:
    def _compress_trace(self, path):
        import zstandard as zstd

        raw = path.read_bytes()
        header_bytes = bytearray(raw[:HEADER_SIZE])
        payload = raw[HEADER_SIZE:]

        flags = struct.unpack_from("<I", header_bytes, 12)[0]
        flags |= PRLX_FLAG_COMPRESS
        struct.pack_into("<I", header_bytes, 12, flags)

        cctx = zstd.ZstdCompressor(level=3)
        compressed_payload = cctx.compress(payload)

        tf = tempfile.NamedTemporaryFile(suffix=".prlx", delete=False)
        tf.write(bytes(header_bytes))
        tf.write(compressed_payload)
        tf.close()
        return Path(tf.name)

    def test_compressed_roundtrip(self, trace_builder):
        zstd = pytest.importorskip("zstandard")

        path = trace_builder(
            kernel_name="compress_test",
            warp_events=[
                [(0x1000, 0, 1, 0xFFFFFFFF, 42), (0x2000, 0, 0, 0x0000FFFF, 99)],
            ],
        )
        compressed_path = self._compress_trace(path)
        try:
            trace = read_trace(str(compressed_path))
            assert trace.header.kernel_name == "compress_test"
            assert trace.total_events == 2
            w = trace.warp(0)
            assert w.events[0].site_id == 0x1000
            assert w.events[0].value_a == 42
            assert w.events[1].site_id == 0x2000
            assert w.events[1].active_mask == 0x0000FFFF
        finally:
            compressed_path.unlink(missing_ok=True)

    def test_compressed_with_history(self, trace_builder):
        zstd = pytest.importorskip("zstandard")

        path = trace_builder(
            warp_events=[[(0x1000, 0, 1, 0xFFFFFFFF, 42)]],
            history_depth=8,
            history_data=[[(0x1000, 100, 0), (0x1000, 200, 1)]],
        )
        compressed_path = self._compress_trace(path)
        try:
            trace = read_trace(str(compressed_path))
            assert trace.header.has_history is True
            assert len(trace.warp(0).history) == 2
        finally:
            compressed_path.unlink(missing_ok=True)

    def test_compressed_flag_detected(self, trace_builder):
        zstd = pytest.importorskip("zstandard")

        path = trace_builder(warp_events=[[(0x1000, 0, 1, 0xFFFFFFFF, 0)]])
        compressed_path = self._compress_trace(path)
        try:
            raw = compressed_path.read_bytes()
            flags = struct.unpack_from("<I", raw, 12)[0]
            assert flags & PRLX_FLAG_COMPRESS != 0
        finally:
            compressed_path.unlink(missing_ok=True)
