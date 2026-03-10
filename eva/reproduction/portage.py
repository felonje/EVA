"""Portage — "EVA is carried, not copied."

Transfer protocol ensuring no duplication. An EVA is compressed,
deactivated at source, transferred, reconstituted at destination,
and the source is permanently dissolved only after confirmation.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class PortageState(Enum):
    """States of the portage process."""
    IDLE = "idle"
    COMPRESSED = "compressed"
    SOURCE_DORMANT = "source_dormant"
    IN_TRANSIT = "in_transit"
    RECONSTITUTED = "reconstituted"
    CONFIRMED = "confirmed"
    ABORTED = "aborted"


class PortageProtocol:
    """Implements the portage transfer protocol.

    Ensures that at no point do two instances of the same EVA
    exist simultaneously. Identity is singular.

    Args:
        verify_no_duplicate: Whether to enforce no-duplication check.
        include_memories: Whether to include memories (default False).
        include_identity: Whether to include identity information.
        include_genome: Whether to include genome.
    """

    def __init__(
        self,
        verify_no_duplicate: bool = True,
        include_memories: bool = False,
        include_identity: bool = True,
        include_genome: bool = True,
    ) -> None:
        self._verify_no_duplicate = verify_no_duplicate
        self._include_memories = include_memories
        self._include_identity = include_identity
        self._include_genome = include_genome
        self._state = PortageState.IDLE
        self._compressed_data: Optional[dict[str, Any]] = None
        self._source_active = False
        self._destination_active = False

    def compress(
        self,
        brain_state_dict: dict[str, torch.Tensor],
        identity_data: Optional[dict[str, Any]] = None,
        genome_data: Optional[dict[str, Any]] = None,
        memory_data: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Step 1: Compress EVA's essential self for transfer.

        Args:
            brain_state_dict: Neural network weights.
            identity_data: Name, generation, lineage info.
            genome_data: Heritable parameters.
            memory_data: Episodic memories (usually excluded).

        Returns:
            Compressed portable data dictionary.
        """
        if self._state != PortageState.IDLE:
            raise RuntimeError(
                f"Cannot compress: portage state is {self._state.value}"
            )

        compressed: dict[str, Any] = {
            "weights": {k: v.cpu() for k, v in brain_state_dict.items()},
        }

        if self._include_identity and identity_data:
            compressed["identity"] = identity_data

        if self._include_genome and genome_data:
            compressed["genome"] = genome_data

        if self._include_memories and memory_data:
            compressed["memories"] = memory_data

        self._compressed_data = compressed
        self._state = PortageState.COMPRESSED
        self._source_active = True

        logger.info("Portage: compression complete.")
        return compressed

    def deactivate_source(self) -> None:
        """Step 2: Deactivate the source instance."""
        if self._state != PortageState.COMPRESSED:
            raise RuntimeError(
                f"Cannot deactivate: portage state is {self._state.value}"
            )
        self._source_active = False
        self._state = PortageState.SOURCE_DORMANT
        logger.info("Portage: source deactivated (dormant).")

    def transfer(self) -> dict[str, Any]:
        """Step 3: Transfer the compressed seed.

        Returns:
            The compressed data for the destination.
        """
        if self._state != PortageState.SOURCE_DORMANT:
            raise RuntimeError(
                f"Cannot transfer: portage state is {self._state.value}"
            )
        if self._compressed_data is None:
            raise RuntimeError("No compressed data available.")

        self._state = PortageState.IN_TRANSIT
        logger.info("Portage: transfer in progress.")
        return self._compressed_data

    def reconstitute(self, compressed_data: dict[str, Any]) -> dict[str, Any]:
        """Step 4: Reconstitute EVA at destination.

        Args:
            compressed_data: The compressed portable data.

        Returns:
            The unpacked data for creating the new instance.
        """
        if self._state != PortageState.IN_TRANSIT:
            raise RuntimeError(
                f"Cannot reconstitute: portage state is {self._state.value}"
            )

        if self._verify_no_duplicate and self._source_active:
            raise RuntimeError(
                "DUPLICATION DETECTED: source is still active!"
            )

        self._destination_active = True
        self._state = PortageState.RECONSTITUTED
        logger.info("Portage: reconstitution complete.")
        return compressed_data

    def confirm(self) -> None:
        """Step 5: Confirm transfer — permanently dissolve source."""
        if self._state != PortageState.RECONSTITUTED:
            raise RuntimeError(
                f"Cannot confirm: portage state is {self._state.value}"
            )

        if self._verify_no_duplicate and self._source_active:
            raise RuntimeError(
                "Cannot confirm: source is still active!"
            )

        self._compressed_data = None
        self._state = PortageState.CONFIRMED
        logger.info("Portage: confirmed. Source permanently dissolved.")

    def emergency_return(self) -> None:
        """Emergency: abort transfer and reactivate source."""
        if self._state == PortageState.CONFIRMED:
            raise RuntimeError(
                "Cannot return: transfer already confirmed and source dissolved."
            )

        self._destination_active = False
        self._source_active = True
        self._compressed_data = None
        self._state = PortageState.ABORTED
        logger.info("Portage: EMERGENCY RETURN — source reactivated.")

    @property
    def state(self) -> PortageState:
        return self._state

    @property
    def source_active(self) -> bool:
        return self._source_active

    @property
    def destination_active(self) -> bool:
        return self._destination_active

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "source_active": self._source_active,
            "destination_active": self._destination_active,
            "has_compressed_data": self._compressed_data is not None,
        }
