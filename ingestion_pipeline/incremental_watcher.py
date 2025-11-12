import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class IncrementalWatcher:
    def __init__(self, watch_dir: Path, state_file: Optional[Path] = None):
        self.watch_dir = Path(watch_dir)
        self.state_file = state_file or self.watch_dir.parent / ".incremental_state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {}

    def _save_state(self) -> None:
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def detect_changes(self) -> dict:
        if not self.watch_dir.exists():
            return {"added": [], "modified": [], "deleted": list(self.state.keys())}

        current_files = {}
        for file_path in self.watch_dir.rglob("*"):
            if file_path.is_file():
                file_hash = self._hash_file(file_path)
                rel_path = str(file_path.relative_to(self.watch_dir))
                current_files[rel_path] = file_hash

        added = [f for f in current_files if f not in self.state]
        modified = [
            f for f in current_files if f in self.state and current_files[f] != self.state[f]
        ]
        deleted = [f for f in self.state if f not in current_files]

        self.state = current_files
        self._save_state()

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
        }

    def _hash_file(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def mark_indexed(self, file_paths: list[str]) -> None:
        for file_path in file_paths:
            if file_path in self.state:
                self.state[file_path] = self._hash_file(self.watch_dir / file_path)
        self._save_state()
