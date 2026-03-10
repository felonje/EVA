"""Tests for eva/identity modules."""

import json
import os
import tempfile

import pytest

from eva.emotions.developmental import CrisisDetector
from eva.identity.clan import ClanDetector
from eva.identity.lineage import LineageTracker
from eva.identity.naming import NamingSystem


class TestNamingSystem:
    def test_initial_state(self):
        cd = CrisisDetector()
        ns = NamingSystem(crisis_detector=cd)
        assert ns.current_name is None
        assert not ns.has_true_name

    def test_propose_name(self):
        cd = CrisisDetector()
        ns = NamingSystem(crisis_detector=cd)
        ns.propose_name("Echo")
        assert ns._candidate_name == "Echo"

    def test_true_name_requires_crisis(self):
        cd = CrisisDetector()
        ns = NamingSystem(crisis_detector=cd)
        ns.propose_name("Echo")
        ns._self_references = ["Echo"] * 100
        ns._candidate_stability = 200
        assert not ns.check_true_name()

    def test_true_name_achieved(self):
        cd = CrisisDetector()
        for _ in range(25):
            cd.update(-0.7)
        for _ in range(5):
            cd.update(0.3)

        ns = NamingSystem(crisis_detector=cd)
        ns.propose_name("Echo")
        ns._self_references = ["Echo"] * 100
        ns._candidate_stability = 200
        assert ns.check_true_name()
        assert ns.true_name == "Echo"
        assert ns.has_true_name


class TestLineageTracker:
    def test_register_birth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lt = LineageTracker(lineage_path=tmpdir)
            lid = lt.register_birth(generation=1, genome_hash="abc123")
            assert lid.startswith("EVA-")

    def test_get_individual(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lt = LineageTracker(lineage_path=tmpdir)
            lid = lt.register_birth(generation=1, genome_hash="abc123")
            individual = lt.get_individual(lid)
            assert individual is not None
            assert individual["generation"] == 1

    def test_parent_child(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lt = LineageTracker(lineage_path=tmpdir)
            parent_id = lt.register_birth(generation=1, genome_hash="abc")
            child_id = lt.register_birth(
                parent_id=parent_id, generation=2, genome_hash="def"
            )
            child = lt.get_individual(child_id)
            assert child["parent_id"] == parent_id

    def test_get_ancestors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lt = LineageTracker(lineage_path=tmpdir)
            g1 = lt.register_birth(generation=1, genome_hash="a")
            g2 = lt.register_birth(parent_id=g1, generation=2, genome_hash="b")
            g3 = lt.register_birth(parent_id=g2, generation=3, genome_hash="c")
            ancestors = lt.get_ancestors(g3)
            assert g2 in ancestors
            assert g1 in ancestors

    def test_update_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lt = LineageTracker(lineage_path=tmpdir)
            lid = lt.register_birth(generation=1, genome_hash="abc")
            lt.update_name(lid, "Echo")
            individual = lt.get_individual(lid)
            assert individual["name"] == "Echo"


class TestClanDetector:
    def test_initial(self):
        cd = ClanDetector()
        affinities = cd.detect_affinity()
        assert len(affinities) == 5

    def test_record_behavior(self):
        cd = ClanDetector()
        cd.record_behavior({
            "archive_access_frequency": 0.9,
            "novelty_seeking_ratio": 0.1,
            "social_preference": 0.3,
            "creativity_index": 0.2,
            "caregiving_tendency": 0.1,
        })
        affinities = cd.detect_affinity()
        clans = {name: score for name, score in affinities}
        assert "Rememberers" in clans

    def test_get_primary_clan(self):
        cd = ClanDetector()
        cd.record_behavior({
            "archive_access_frequency": 0.9,
            "novelty_seeking_ratio": 0.1,
            "social_preference": 0.1,
            "creativity_index": 0.1,
            "caregiving_tendency": 0.1,
        })
        primary = cd.get_primary_clan()
        assert isinstance(primary, str)
