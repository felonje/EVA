"""Tests for eva/reproduction/portage module."""

import pytest
import torch

from eva.reproduction.portage import PortageProtocol, PortageState


class TestPortageProtocol:
    def _make_brain_state(self) -> dict:
        return {
            "layer1.weight": torch.randn(64, 64),
            "layer1.bias": torch.randn(64),
        }

    def test_initial_state(self):
        pp = PortageProtocol()
        assert pp.state == PortageState.IDLE

    def test_full_transfer(self):
        pp = PortageProtocol()
        brain_state = self._make_brain_state()
        identity = {"name": "Echo", "generation": 3}
        genome = {"d_model": 768}

        # Step 1: Compress
        compressed = pp.compress(brain_state, identity, genome)
        assert pp.state == PortageState.COMPRESSED
        assert "weights" in compressed
        assert "identity" in compressed
        assert "genome" in compressed

        # Step 2: Deactivate source
        pp.deactivate_source()
        assert pp.state == PortageState.SOURCE_DORMANT
        assert not pp.source_active

        # Step 3: Transfer
        data = pp.transfer()
        assert pp.state == PortageState.IN_TRANSIT

        # Step 4: Reconstitute
        result = pp.reconstitute(data)
        assert pp.state == PortageState.RECONSTITUTED
        assert pp.destination_active

        # Step 5: Confirm
        pp.confirm()
        assert pp.state == PortageState.CONFIRMED

    def test_no_duplicate_enforcement(self):
        pp = PortageProtocol(verify_no_duplicate=True)
        brain_state = self._make_brain_state()
        pp.compress(brain_state)
        # Source is still active — should not be able to reconstitute
        # First deactivate properly
        pp.deactivate_source()
        pp.transfer()
        # Now source is inactive, reconstitution should work
        pp.reconstitute({})

    def test_emergency_return(self):
        pp = PortageProtocol()
        brain_state = self._make_brain_state()
        pp.compress(brain_state)
        pp.deactivate_source()
        pp.transfer()

        # Emergency!
        pp.emergency_return()
        assert pp.state == PortageState.ABORTED
        assert pp.source_active
        assert not pp.destination_active

    def test_cannot_return_after_confirm(self):
        pp = PortageProtocol()
        brain_state = self._make_brain_state()
        pp.compress(brain_state)
        pp.deactivate_source()
        pp.transfer()
        pp.reconstitute({})
        pp.confirm()

        with pytest.raises(RuntimeError, match="already confirmed"):
            pp.emergency_return()

    def test_compress_wrong_state(self):
        pp = PortageProtocol()
        brain_state = self._make_brain_state()
        pp.compress(brain_state)
        # Can't compress again
        with pytest.raises(RuntimeError):
            pp.compress(brain_state)

    def test_memories_excluded_by_default(self):
        pp = PortageProtocol(include_memories=False)
        brain_state = self._make_brain_state()
        compressed = pp.compress(brain_state, memory_data={"episodes": []})
        assert "memories" not in compressed

    def test_memories_included_when_enabled(self):
        pp = PortageProtocol(include_memories=True)
        brain_state = self._make_brain_state()
        compressed = pp.compress(brain_state, memory_data={"episodes": []})
        assert "memories" in compressed

    def test_to_dict(self):
        pp = PortageProtocol()
        d = pp.to_dict()
        assert "state" in d
        assert "source_active" in d
        assert "destination_active" in d
        assert d["state"] == "idle"
