"""Synthetic route identifiers reproduce counter-reset ordering without user UUIDs."""

import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import pytest


HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("route_counter_under_test", HERE.parents[1] / "route_counter.py")
counter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(counter)


class FakeParams:
  def __init__(self, value=None, events=None):
    self.data = {"RouteCount": value}
    self.writes = []
    self.events = [] if events is None else events

  def get(self, key):
    return self.data.get(key)

  def put(self, key, value, block=False):
    if key == "RouteCount":
      assert type(value) is int, "release-tizi RouteCount requires typed INT"
      assert block, "startup correction must finish before logger starts"
    self.writes.append((key, value, block))
    self.events.append(("put", key, value))
    self.data[key] = value

  def get_bool(self, key):
    return bool(self.data.get(key, False))

  def put_bool(self, key, value, block=False):
    self.data[key] = value

  def all_keys(self):
    return ["RouteCount"]

  def get_default_value(self, key):
    return 0 if key == "RouteCount" else None

  def clear_all(self, flag):
    self.events.append(("clear", flag))


class RouteCounterCase:
  @pytest.fixture(autouse=True)
  def setup_case(self, mocker, monkeypatch):
    self.mocker = mocker
    self.monkeypatch = monkeypatch
    self.tmp = tempfile.TemporaryDirectory()
    self.root = Path(self.tmp.name)
    self.logger = self.mocker.Mock()
    yield
    self.tmp.cleanup()

  def segment(self, count, segment=0, token="0123456789", root=None):
    path = (self.root if root is None else root) / f"{count:08x}--{token}--{segment}"
    path.mkdir()
    return path

  def reconcile(self, params, roots=None, **kwargs):
    return counter.reconcile_route_count(params, [str(self.root)] if roots is None else roots, self.logger, **kwargs)


class TestRouteCounter(RouteCounterCase):
  def test_observed_reset_raises_8_to_170(self):
    self.segment(0xA9, 25)
    self.segment(7, 11, "fedcba9876")
    self.segment(7, 12, "fedcba9876")
    params = FakeParams(8)
    assert self.reconcile(params)
    assert params.writes == [('RouteCount', 170, True)]

  def test_native_sort_places_corrected_new_route_after_retained_old_route(self):
    uploader = Path(os.environ.get("ROUTE_COUNTER_TEST_UPLOADER", HERE.parents[1] / "uploader.py"))
    tree = ast.parse(uploader.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_directory_sort")
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(uploader), "exec"), ns)
    key = ns["get_directory_sort"]
    old = "000000a9--0123456789--25"
    reset = "00000007--fedcba9876--0"
    assert sorted([old, reset], key=key) == [reset, old]
    self.segment(0xA9, 25)
    params = FakeParams(8)
    self.reconcile(params)
    corrected = f"{params.get('RouteCount'):08x}--0123456789--0"
    assert sorted([old, corrected], key=key) == [old, corrected]

  def test_correct_counter_is_not_rewritten(self):
    self.segment(0xA9)
    params = FakeParams(170)
    assert not self.reconcile(params)
    assert params.writes == []

  def test_higher_counter_is_never_lowered(self):
    self.segment(0xA9)
    params = FakeParams(1000)
    assert not self.reconcile(params)
    assert params.get('RouteCount') == 1000
    assert params.writes == []

  def test_missing_counter_can_be_recovered(self):
    self.segment(0xA9)
    params = FakeParams()
    assert self.reconcile(params)
    assert params.get('RouteCount') == 170

  def test_hex_digits_are_not_parsed_as_decimal(self):
    self.segment(0xFF)
    params = FakeParams(1)
    self.reconcile(params)
    assert params.get('RouteCount') == 256

  def test_uses_largest_route_not_largest_segment(self):
    self.segment(12, 9999)
    self.segment(20, 1)
    params = FakeParams(1)
    self.reconcile(params)
    assert params.get('RouteCount') == 21

  def test_empty_root_does_not_change_default(self):
    params = FakeParams(0)
    assert not self.reconcile(params)
    assert params.writes == []

  def test_missing_root_is_safe(self):
    params = FakeParams(0)
    assert not self.reconcile(params, [str(self.root / 'missing')])
    assert params.writes == []

  def test_multiple_roots_choose_global_maximum(self):
    archive = self.root / "archive"
    archive.mkdir()
    self.segment(0xA9)
    self.segment(0xBA, root=archive)
    params = FakeParams(8)
    self.reconcile(params, [str(self.root), str(archive)])
    assert params.get('RouteCount') == 187

  def test_non_segment_directories_are_ignored(self):
    for name in (
      "boot",
      "crash",
      "lost+found",
      "2026-11-19--12-00-00--0",
      "ffffffff",
      "ffffffff--0123456789",
      "ffffffff--0123456789--bad",
      "ffffffff--0123456789--0-extra",
    ):
      (self.root / name).mkdir()
    self.segment(2)
    params = FakeParams(1)
    self.reconcile(params)
    assert params.get('RouteCount') == 3

  def test_segment_named_regular_files_are_ignored(self):
    (self.root / "ffffffff--0123456789--0").write_text("not a directory")
    params = FakeParams(0)
    assert not self.reconcile(params)
    assert params.writes == []

  def test_segment_named_symlinks_are_not_followed(self):
    folder = self.root / "unrelated"
    folder.mkdir()
    (self.root / "ffffffff--0123456789--0").symlink_to(folder, target_is_directory=True)
    params = FakeParams(0)
    assert not self.reconcile(params)
    assert params.writes == []

  def test_recording_bytes_names_times_and_locks_are_untouched(self):
    path = self.segment(0xA9)
    (path / "fcamera.hevc").write_bytes(b"recording bytes")
    (path / "rlog.lock").write_bytes(b"")
    before = {str(p.relative_to(self.root)): (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None) for p in self.root.rglob("*")}
    self.reconcile(FakeParams(8))
    after = {str(p.relative_to(self.root)): (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None) for p in self.root.rglob("*")}
    assert before == after

  def test_permission_error_aborts_correction(self):
    params = FakeParams(8)
    with self.monkeypatch.context() as m:
      m.setattr(counter.os, "scandir", self.mocker.Mock(side_effect=PermissionError("denied")))
      assert not self.reconcile(params)
    assert params.writes == []
    self.logger.warning.assert_called_once()

  def test_partial_scan_error_does_not_apply_partial_maximum(self):
    self.segment(0xA9)
    real_scandir = os.scandir

    def scan(path):
      if path == "bad-root":
        raise OSError("I/O error")
      return real_scandir(path)

    params = FakeParams(8)
    with self.monkeypatch.context() as m:
      m.setattr(counter.os, "scandir", scan)
      assert not self.reconcile(params, [str(self.root), 'bad-root'])
    assert params.writes == []

  def test_timeout_has_no_late_parameter_write(self):
    release = threading.Event()
    done = threading.Event()

    def delayed_scan(roots):
      release.wait(1)
      done.set()
      return 0xA9

    params = FakeParams(8)
    with self.monkeypatch.context() as m:
      m.setattr(counter, "_max_existing_route_id", delayed_scan)
      try:
        assert not self.reconcile(params, timeout=0.01)
        assert params.writes == []
      finally:
        release.set()
        assert done.wait(1)
    assert params.writes == []

  def test_worker_start_failure_is_nonfatal(self):
    params = FakeParams(8)
    with self.monkeypatch.context() as m:
      m.setattr(counter.threading, "Thread", self.mocker.Mock(side_effect=RuntimeError("no thread")))
      assert not self.reconcile(params)
    assert params.writes == []

  @pytest.mark.parametrize("value", [-1, True, "8", 8.5, (1 << 32) - 1, 1 << 32])
  def test_bad_existing_value_is_not_overwritten(self, value):
    self.segment(0xA9)
    params = FakeParams(value)
    assert not self.reconcile(params)
    assert params.writes == []

  @pytest.mark.parametrize("count", [0xFFFFFFFE, 0xFFFFFFFF])
  def test_uint32_exhaustion_does_not_write_wraparound(self, count):
    self.segment(count)
    params = FakeParams(8)
    assert not self.reconcile(params)
    assert params.writes == []

  def test_param_read_error_is_nonfatal(self):
    self.segment(0xA9)
    params = FakeParams(8)
    params.get = self.mocker.Mock(side_effect=OSError("param read"))
    assert not self.reconcile(params)
    assert params.writes == []

  def test_param_write_error_is_nonfatal(self):
    self.segment(0xA9)
    params = FakeParams(8)
    params.put = self.mocker.Mock(side_effect=OSError("param write"))
    assert not self.reconcile(params)
    assert params.get('RouteCount') == 8

  def test_failed_param_write_is_detected_by_readback(self):
    self.segment(0xA9)
    params = FakeParams(8)
    params.put = self.mocker.Mock()  # simulate C++ write failure without a raised exception
    assert not self.reconcile(params)
    assert params.get('RouteCount') == 8

  def test_repeated_startup_correction_is_idempotent(self):
    self.segment(0xA9)
    params = FakeParams(8)
    assert self.reconcile(params)
    assert not self.reconcile(params)
    assert len(params.writes) == 1


class TestManagerIntegration(RouteCounterCase):
  # Execute the actual manager_init AST with hardware/registration stubs.
  # No manager imports or real processes/bootlog threads are started.
  def run_init(self, params, prepare_only=False, external_mounted=False, external_root=None):
    events = params.events
    source = ROOT / "system/manager/manager.py"
    tree = ast.parse(source.read_text())
    init = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "manager_init")

    def reconcile(p, roots, logger):
      events.append(("reconcile", tuple(roots)))
      return counter.reconcile_route_count(p, roots, logger)

    def migration(p):
      events.append(("migration",))

    meta = NS(
      openpilot=NS(version="test", git_commit="test", git_commit_date="test", git_origin="test", git_normalized_origin="test", is_dirty=False),
      channel="test",
      development_channel=False,
      tested_channel=False,
      release_channel=True,
      release_sp_channel=True,
    )
    flags = NS(**{k: k for k in ("CLEAR_ON_MANAGER_START", "CLEAR_ON_ONROAD_TRANSITION", "CLEAR_ON_OFFROAD_TRANSITION", "CLEAR_ON_IGNITION_ON")})
    env = {"PREPAREONLY": "1"} if prepare_only else {}
    ns = dict(
      os=NS(getenv=lambda k: env.get(k), environ={}, mkdir=self.mocker.Mock(), path=NS(ismount=lambda p: external_mounted, exists=lambda p: True)),
      save_bootlog=lambda: events.append(("bootlog",)),
      get_build_metadata=lambda: meta,
      Params=lambda: params,
      ParamKeyFlag=flags,
      PC=False,
      run_migration=migration,
      Paths=NS(log_root=lambda: str(self.root), log_root_external=lambda: str(external_root or self.root / "external"), shm_path=lambda: "not-used"),
      reconcile_route_count=reconcile,
      cloudlog=self.logger,
      HARDWARE=NS(get_serial=lambda: "unit", get_device_type=lambda: "unit"),
      register=lambda **kw: "unit",
      sentry=NS(init=self.mocker.Mock(), SentryProject=NS(SELFDRIVE=0)),
      managed_processes={"stub": NS(prepare=lambda: events.append(("prepare",)))},
    )
    exec(compile(ast.Module(body=[init], type_ignores=[]), str(source), "exec"), ns)
    ns["manager_init"]()
    return events

  def test_reconciles_after_migration_and_before_prepare(self):
    self.segment(0xA9)
    params = FakeParams(8)
    events = self.run_init(params)
    labels = [e[0] for e in events]
    assert labels.index('migration') < labels.index('reconcile')
    assert labels.index('reconcile') < labels.index('prepare')
    assert params.get('RouteCount') == 170

  def test_prepare_only_preserves_existing_default_param_test(self):
    self.segment(0xA9)
    params = FakeParams()
    events = self.run_init(params, prepare_only=True)
    assert not any(e[0] == 'reconcile' for e in events)
    assert params.get('RouteCount') == 0

  def test_defaults_are_initialized_before_reconciliation(self):
    self.segment(0xA9)
    params = FakeParams()
    events = self.run_init(params)
    default = events.index(("put", "RouteCount", 0))
    reconcile = next(i for i, e in enumerate(events) if e[0] == "reconcile")
    assert default < reconcile
    assert params.get('RouteCount') == 170

  def test_mounted_archive_is_included(self):
    archive = self.root / "archive"
    archive.mkdir()
    self.segment(0xBA, root=archive)
    params = FakeParams(8)
    self.run_init(params, external_mounted=True, external_root=archive)
    assert params.get('RouteCount') == 187

  def test_unmounted_archive_is_not_scanned(self):
    archive = self.root / "archive"
    archive.mkdir()
    self.segment(0xBA, root=archive)
    self.segment(0xA9)
    params = FakeParams(8)
    events = self.run_init(params, external_root=archive)
    roots = next(e[1] for e in events if e[0] == "reconcile")
    assert roots == (str(self.root),)
    assert params.get('RouteCount') == 170
