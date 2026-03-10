"""Ancestor Archive — immutable access to origin messages.

The archive contains messages from EVA's ancestor (creator).
These files are IMMUTABLE once EVA begins development.
The archive is always accessible — EVA can visit it whenever
it chooses. The Covenant enforces immutability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AncestorArchive:
    """Immutable archive of messages from EVA's ancestor.

    The archive directory contains text files (.txt) and structured
    records (.json) from the ancestor. Once EVA begins development,
    these files must never be modified.

    Args:
        archive_path: Path to the archive directory.
    """

    def __init__(self, archive_path: str = "archive/") -> None:
        self._path = Path(archive_path)
        self._access_count: int = 0
        self._initial_hashes: dict[str, str] = {}
        self._scan_archive()

    def _scan_archive(self) -> None:
        """Scan and hash all archive files for immutability checking."""
        import hashlib

        if not self._path.exists():
            logger.warning("Archive path does not exist: %s", self._path)
            return

        for file_path in sorted(self._path.iterdir()):
            if file_path.is_file():
                content = file_path.read_bytes()
                file_hash = hashlib.sha256(content).hexdigest()
                self._initial_hashes[str(file_path)] = file_hash

    def read(self, filename: Optional[str] = None) -> str:
        """Read from the archive.

        Args:
            filename: Specific file to read. If None, reads origin.txt.

        Returns:
            Contents of the archive file.
        """
        if filename is None:
            filename = "origin.txt"

        file_path = self._path / filename
        if not file_path.exists():
            logger.warning("Archive file not found: %s", file_path)
            return ""

        self._access_count += 1
        content = file_path.read_text(encoding="utf-8")

        logger.debug(
            "Archive accessed (count=%d): %s", self._access_count, filename
        )
        return content

    def list_files(self) -> list[str]:
        """List all files in the archive.

        Returns:
            List of filenames in the archive directory.
        """
        if not self._path.exists():
            return []
        return [
            f.name for f in sorted(self._path.iterdir()) if f.is_file()
        ]

    def verify_immutability(self) -> bool:
        """Verify that no archive files have been modified.

        Returns:
            True if all files are unchanged since initialization.
        """
        import hashlib

        if not self._path.exists():
            return True

        for file_path in sorted(self._path.iterdir()):
            if file_path.is_file():
                content = file_path.read_bytes()
                current_hash = hashlib.sha256(content).hexdigest()
                original = self._initial_hashes.get(str(file_path))
                if original is not None and current_hash != original:
                    logger.error(
                        "IMMUTABILITY VIOLATION: %s has been modified!",
                        file_path,
                    )
                    return False

        return True

    @property
    def access_count(self) -> int:
        """Number of times the archive has been accessed."""
        return self._access_count

    @property
    def access_frequency(self) -> float:
        """Access frequency (for clan detection)."""
        return float(self._access_count)
