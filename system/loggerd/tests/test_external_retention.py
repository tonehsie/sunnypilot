"""Exercise actual deletion functions with fake storage and mocked file mutations."""

import ast
from pathlib import Path
import posixpath
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[3]
LOG_ROOT = "/data/media/0/realdata"
ARCHIVE_ROOT = "/mnt/external_realdata"
OLD = "00000001--0123456789--0"
CURRENT = "00000002--fedcba9876--0"


class Storage:
  def __init__(self, mocker, root=LOG_ROOT, external=True, archive=False, pc=False):
    self.root = root
    self.pc = pc
    self.mounts = {root} if external else set()
    if archive:
      self.mounts.add(ARCHIVE_ROOT)
    self.devices = {"/data": 1, root: 2 if external else 1, ARCHIVE_ROOT: 3}
    self.percent = {root: 3.0, ARCHIVE_ROOT: 3.0, "/data": 40.0}
    self.bytes = {root: 20 * 1024**3, ARCHIVE_ROOT: 20 * 1024**3}
    self.stat_error = False
    self.vfs_error = False
    self.realpath_error = False
    self.aliases = {}
    self.locked = {CURRENT}
    self.preserved = set()
    self.dirs = {root: [OLD, CURRENT], ARCHIVE_ROOT: [OLD]}
    self.logger = mocker.Mock()
    self.shutil = NS(rmtree=mocker.Mock(), move=mocker.Mock())
    self.percent_calls = []
    self.byte_calls = []

  def realpath(self, path):
    if self.realpath_error:
      raise OSError("path lookup failed")
    path = posixpath.normpath(path)
    return self.aliases.get(path, path)

  def stat(self, path):
    if self.stat_error:
      raise OSError("device lookup failed")
    return NS(st_dev=self.devices[path])

  def available_percent(self, default, path_type="internal"):
    self.percent_calls.append((path_type, default))
    if self.vfs_error:
      return default
    return self.percent[self.root if path_type == "internal" else ARCHIVE_ROOT]

  def available_bytes(self, default, path_type="internal"):
    self.byte_calls.append((path_type, default))
    if self.vfs_error:
      return default
    return self.bytes[self.root if path_type == "internal" else ARCHIVE_ROOT]

  def load(self):
    path = ROOT / "system/loggerd/deleter.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))]
    ns = dict(
      os=NS(
        path=NS(realpath=self.realpath, join=posixpath.join),
        stat=self.stat,
        listdir=lambda p: ["rlog.lock"] if posixpath.basename(p) in self.locked else ["rlog.zst"],
      ),
      time=NS(monotonic=lambda: 1.0),
      shutil=self.shutil,
      threading=NS(Event=object),
      AGNOS=not self.pc,
      Path=lambda p: NS(is_mount=lambda: p in self.mounts),
      Paths=NS(log_root=lambda: self.root, log_root_external=lambda: ARCHIVE_ROOT),
      cloudlog=self.logger,
      get_available_percent=self.available_percent,
      get_available_bytes=self.available_bytes,
      listdir_by_creation=lambda p: self.dirs[p],
      getxattr=lambda p, key: b"1" if posixpath.basename(p) in self.preserved else None,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns

  def run(self):
    waits = []
    checks = iter([False, True])
    event = NS(is_set=lambda: next(checks), wait=lambda t: waits.append(t))
    ns = self.load()
    ns["deleter_thread"](event)
    return ns, waits


@pytest.mark.parametrize("path", [LOG_ROOT, LOG_ROOT + "/"])
def test_known_independent_mount_gets_two_percent(mocker, path):
  storage = Storage(mocker, archive=True)
  assert storage.load()["min_available_percent"](path) == 2


def test_unmounted_root_keeps_internal_reserve(mocker):
  storage = Storage(mocker, external=False)
  assert storage.load()["min_available_percent"](LOG_ROOT) == 10


def test_same_device_mount_keeps_internal_reserve(mocker):
  storage = Storage(mocker)
  storage.devices[LOG_ROOT] = 1
  assert storage.load()["min_available_percent"](LOG_ROOT) == 10


@pytest.mark.parametrize("root", ["/", "/data", "/tmp/logs", "/home/comma/logs"])
def test_unknown_roots_are_not_assumed_external(mocker, root):
  storage = Storage(mocker, root=root)
  assert storage.load()["min_available_percent"](root) == 10


def test_symlink_to_unknown_destination_is_conservative(mocker):
  storage = Storage(mocker)
  storage.aliases[LOG_ROOT] = "/somewhere/else"
  assert storage.load()["min_available_percent"](LOG_ROOT) == 10


def test_pc_keeps_original_policy(mocker):
  storage = Storage(mocker, pc=True)
  assert storage.load()["min_available_percent"](LOG_ROOT) == 10


@pytest.mark.parametrize("error", ["stat_error", "realpath_error"])
def test_detection_errors_keep_internal_policy(mocker, error):
  storage = Storage(mocker)
  setattr(storage, error, True)
  assert storage.load()["min_available_percent"](LOG_ROOT) == 10


@pytest.mark.parametrize("free", [10.0064, 7.0, 3.0, 2.0])
def test_external_above_or_at_two_percent_is_not_cleaned(mocker, free):
  storage = Storage(mocker)
  storage.percent[LOG_ROOT] = free
  _, waits = storage.run()
  storage.shutil.rmtree.assert_not_called()
  assert waits == [30]
  assert storage.percent_calls[0] == ("internal", 3)


@pytest.mark.parametrize("free", [1.9999, 1.5, 0.0])
def test_external_below_two_percent_is_cleaned(mocker, free):
  storage = Storage(mocker)
  storage.percent[LOG_ROOT] = free
  _, waits = storage.run()
  storage.shutil.rmtree.assert_called_once_with(LOG_ROOT + "/" + OLD)
  assert waits == [0.1]


def test_five_gib_floor_remains_even_above_two_percent(mocker):
  storage = Storage(mocker)
  storage.percent[LOG_ROOT] = 5.0
  storage.bytes[LOG_ROOT] = 5 * 1024**3 - 1
  ns, _ = storage.run()
  assert ns["MIN_BYTES"] == 5 * 1024**3
  storage.shutil.rmtree.assert_called_once_with(LOG_ROOT + "/" + OLD)


def test_exactly_five_gib_is_not_below_floor(mocker):
  storage = Storage(mocker)
  storage.bytes[LOG_ROOT] = 5 * 1024**3
  storage.run()
  storage.shutil.rmtree.assert_not_called()


def test_active_lock_is_still_protected(mocker):
  storage = Storage(mocker)
  storage.percent[LOG_ROOT] = 1.0
  storage.locked = {OLD, CURRENT}
  storage.run()
  storage.shutil.rmtree.assert_not_called()


def test_preserved_segment_still_has_lower_deletion_priority(mocker):
  storage = Storage(mocker)
  storage.percent[LOG_ROOT] = 1.0
  storage.preserved = {OLD}
  storage.locked = set()
  storage.run()
  storage.shutil.rmtree.assert_called_once_with(LOG_ROOT + "/" + CURRENT)


def test_space_read_failure_does_not_trigger_delete(mocker):
  storage = Storage(mocker)
  storage.vfs_error = True
  storage.run()
  storage.shutil.rmtree.assert_not_called()


@pytest.mark.parametrize("free", [11.0, 10.0, 9.9999, 3.0])
def test_internal_behavior_matches_original(mocker, free):
  actual = Storage(mocker, external=False)
  actual.percent[LOG_ROOT] = free
  _, waits = actual.run()
  if free < 10:
    actual.shutil.rmtree.assert_called_once_with(LOG_ROOT + "/" + OLD)
    assert waits == [0.1]
  else:
    actual.shutil.rmtree.assert_not_called()
    assert waits == [30]


def test_unmounted_external_falls_back_to_ten_percent(mocker):
  storage = Storage(mocker)
  storage.mounts.clear()
  storage.devices[LOG_ROOT] = 1
  storage.percent[LOG_ROOT] = 9.0
  storage.run()
  storage.shutil.rmtree.assert_called_once_with(LOG_ROOT + "/" + OLD)


@pytest.mark.parametrize("archive_free,deleted", [(11.0, False), (10.0, False), (9.9999, True), (3.0, True)])
def test_native_archive_keeps_original_policy(mocker, archive_free, deleted):
  storage = Storage(mocker, external=False, archive=True)
  storage.percent[LOG_ROOT] = 9.0
  storage.percent[ARCHIVE_ROOT] = archive_free
  storage.run()
  assert storage.shutil.rmtree.called == deleted
  storage.shutil.move.assert_called_once_with(LOG_ROOT + "/" + OLD, ARCHIVE_ROOT + "/" + OLD)


def test_archive_on_data_filesystem_keeps_ten_percent(mocker):
  storage = Storage(mocker, external=False, archive=True)
  storage.devices[ARCHIVE_ROOT] = 1
  storage.percent[LOG_ROOT] = 9.0
  storage.percent[ARCHIVE_ROOT] = 9.0
  storage.run()
  storage.shutil.rmtree.assert_called_once_with(ARCHIVE_ROOT + "/" + OLD)


def load_config(agnos, log_root, system_free=12, recording_free=1.5):
  path = ROOT / "system/loggerd/config.py"
  nodes = [n for n in ast.parse(path.read_text()).body if not isinstance(n, (ast.Import, ast.ImportFrom))]

  def statvfs(path):
    free = system_free if path == "/data" else recording_free
    return NS(f_bavail=int(free * 100), f_blocks=10000, f_frsize=4096)

  ns = dict(AGNOS=agnos, os=NS(statvfs=statvfs), Paths=NS(log_root=lambda: log_root, log_root_external=lambda: ARCHIVE_ROOT))
  exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
  return ns


def test_agnos_tracks_system_and_recordings_separately():
  config = load_config(True, LOG_ROOT)
  available = config["get_available_percent"]
  assert available(100.0, path_type="system") == 12
  assert available(100.0) == 1.5
  assert available(100.0, path_type="external") == 1.5


def test_non_agnos_system_measurement_uses_original_log_path():
  config = load_config(False, "/home/user/logs")
  assert config["get_available_percent"](100.0, path_type="system") == 1.5


@pytest.mark.parametrize("system_free", [1.9, 2, 6.9, 7, 10])
@pytest.mark.parametrize("recording_free", [1.9, 2, 20])
def test_hardwared_publishes_internal_free_space_without_changing_startup_gate(system_free, recording_free):
  config = load_config(True, LOG_ROOT, system_free, recording_free)
  path = ROOT / "system/hardware/hardwared.py"
  thread = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "hardware_thread")
  nodes = [n for n in ast.walk(thread) if isinstance(n, ast.Assign)]
  publish = next(n for n in nodes if ast.unparse(n.targets[0]) == "msg.deviceState.freeSpacePercent" and isinstance(n.value, ast.Call))
  gate = next(n for n in nodes if ast.unparse(n.targets[0]) == "startup_conditions['free_space']")
  state = NS()
  ns = dict(get_available_percent=config["get_available_percent"], msg=NS(deviceState=state), startup_conditions={})
  exec(compile(ast.Module(body=[publish, gate], type_ignores=[]), str(path), "exec"), ns)
  assert state.freeSpacePercent == system_free
  assert ns["startup_conditions"]["free_space"] == (system_free > 2)
