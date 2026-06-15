"""Tests for PRLX session functionality."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from .conftest import make_trace


class TestSessionAutoDetect:
    def test_session_auto_detect_directories(self, tmp_path):
        """cmd_diff should auto-set session=True when both paths are directories."""
        from prlx.cli import cmd_diff

        session_a = tmp_path / "sess_a"
        session_b = tmp_path / "sess_b"
        session_a.mkdir()
        session_b.mkdir()

        # cmd_diff calls the differ binary via subprocess, mock it
        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = "/usr/bin/false"
            with mock.patch("subprocess.call") as mock_call:
                mock_call.return_value = 0

                args = SimpleNamespace(
                    trace_a=str(session_a),
                    trace_b=str(session_b),
                    map=None,
                    values=False,
                    verbose=False,
                    limit=None,
                    lookahead=None,
                    max_shown=10,
                    tui=False,
                    float=False,
                    force=False,
                    session=False,  # Not explicitly set
                )

                cmd_diff(args)

                # Verify --session was passed to the differ binary
                call_args = mock_call.call_args[0][0]
                assert "--session" in call_args


class TestSessionInspect:
    def test_session_inspect_manifest(self, tmp_path, capsys):
        """session inspect should print formatted table from session.json."""
        from prlx.cli import cmd_session_inspect

        session_dir = tmp_path / "my_session"
        session_dir.mkdir()

        manifest = [
            {"kernel": "scale_kernel", "launch": 0,
             "file": "scale_0.prlx", "grid": [4, 1, 1], "block": [256, 1, 1]},
            {"kernel": "reduce_kernel", "launch": 1,
             "file": "reduce_1.prlx", "grid": [1, 1, 1], "block": [128, 1, 1]},
        ]
        (session_dir / "session.json").write_text(json.dumps(manifest))

        args = SimpleNamespace(session_dir=str(session_dir))
        result = cmd_session_inspect(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "scale_kernel" in captured.out
        assert "reduce_kernel" in captured.out
        assert "Launches: 2" in captured.out
        assert "4x1x1" in captured.out
        assert "256x1x1" in captured.out

    def test_session_inspect_missing_manifest(self, tmp_path, capsys):
        """session inspect should error when no session.json exists."""
        from prlx.cli import cmd_session_inspect

        args = SimpleNamespace(session_dir=str(tmp_path))
        result = cmd_session_inspect(args)
        assert result == 1
        assert "No session.json" in capsys.readouterr().err


class TestSessionDiffPassthrough:
    def test_diff_traces_session_passthrough(self, tmp_path):
        """session diff should delegate to cmd_diff with session=True."""
        from prlx.cli import cmd_session_diff

        session_a = tmp_path / "sess_a"
        session_b = tmp_path / "sess_b"
        session_a.mkdir()
        session_b.mkdir()

        with mock.patch("prlx.cli.cmd_diff") as mock_diff:
            mock_diff.return_value = 0

            args = SimpleNamespace(
                session_a=str(session_a),
                session_b=str(session_b),
            )

            cmd_session_diff(args)

            # cmd_diff should have been called
            mock_diff.assert_called_once()
            diff_args = mock_diff.call_args[0][0]
            assert diff_args.session is True
            assert diff_args.trace_a == str(session_a)
            assert diff_args.trace_b == str(session_b)


class TestSessionImport:
    def test_session_context_manager_import(self):
        """prlx module should be importable without errors."""
        import prlx
        # session-related CLI functions should be accessible
        from prlx.cli import cmd_session, cmd_session_capture, cmd_session_inspect, cmd_session_diff
        assert callable(cmd_session)
        assert callable(cmd_session_capture)
        assert callable(cmd_session_inspect)
        assert callable(cmd_session_diff)
