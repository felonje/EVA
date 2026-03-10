"""Tests for eva/emotions modules."""

import pytest

from eva.emotions.affect import AffectiveState
from eva.emotions.developmental import CrisisDetector, DevelopmentalEmotions
from eva.emotions.homeostasis import Homeostasis
from eva.emotions.modulation import EmotionalModulation


class TestAffectiveState:
    def test_initial_state(self):
        state = AffectiveState()
        assert state.valence == 0.0
        assert state.arousal == 0.5
        assert state.dominance == 0.5

    def test_update(self):
        state = AffectiveState()
        state.update(
            prediction_success=1.0,
            prediction_error=0.1,
            action_success=0.8,
            caregiver_recency=0.9,
            caregiver_contingency=0.8,
        )
        assert state.valence > 0.0

    def test_circuit_breakers(self):
        state = AffectiveState()
        state.valence = -1.0

        class MockCfg:
            valence_floor = -0.8
            arousal_ceiling = 0.95

        state.apply_circuit_breakers(MockCfg())
        assert state.valence >= -0.8

    def test_get_vector(self):
        state = AffectiveState()
        vec = state.get_vector()
        assert len(vec) == 5

    def test_to_dict(self):
        state = AffectiveState()
        d = state.to_dict()
        assert "valence" in d
        assert "arousal" in d
        assert "dominance" in d
        assert "novelty_feeling" in d
        assert "social" in d


class TestDevelopmentalEmotions:
    def test_detect_empty_config(self):
        dev = DevelopmentalEmotions({})
        state = AffectiveState()
        result = dev.detect(state)
        assert isinstance(result, list)

    def test_detect_with_config(self):
        config = {
            "wonder": {
                "region": {
                    "valence": [0.3, 1.0],
                    "arousal": [0.3, 0.8],
                    "dominance": [0.3, 1.0],
                    "novelty_feeling": [0.6, 1.0],
                    "social": [-1.0, 1.0],
                },
                "danger": "obsession",
                "breaker": "perseveration_limit",
            }
        }
        dev = DevelopmentalEmotions(config)
        state = AffectiveState()
        state.valence = 0.5
        state.arousal = 0.5
        state.dominance = 0.5
        state.novelty_feeling = 0.8
        result = dev.detect(state)
        emotions_found = [name for name, _ in result]
        assert "wonder" in emotions_found


class TestCrisisDetector:
    def test_no_crisis(self):
        cd = CrisisDetector()
        for _ in range(10):
            cd.update(0.5)
        assert not cd.crisis_survived()

    def test_crisis_survived(self):
        cd = CrisisDetector()
        for _ in range(25):
            cd.update(-0.7)
        for _ in range(5):
            cd.update(0.3)
        assert cd.crisis_survived()

    def test_crisis_count(self):
        cd = CrisisDetector()
        for _ in range(25):
            cd.update(-0.7)
        for _ in range(5):
            cd.update(0.3)
        for _ in range(25):
            cd.update(-0.6)
        for _ in range(5):
            cd.update(0.2)
        assert cd.crises_survived >= 2


class TestHomeostasis:
    def test_initial_drives(self):
        h = Homeostasis()
        drives = h.get_drives()
        assert "curiosity_hunger" in drives
        assert "rest_need" in drives
        assert "social_need" in drives

    def test_rest_need_increases(self):
        h = Homeostasis()
        for i in range(100):
            h.update(curiosity_reward=0.5, steps_active=i, steps_since_social=0)
        assert h.rest_need > 0.0

    def test_needs_rest(self):
        h = Homeostasis()
        h.rest_need = 0.9
        assert h.needs_rest()

    def test_needs_social(self):
        h = Homeostasis()
        h.social_need = 0.9
        assert h.needs_social()


class TestEmotionalModulation:
    def test_learning_rate_multiplier(self):
        mod = EmotionalModulation()
        affect = AffectiveState()
        h = Homeostasis()
        lr_mult = mod.get_learning_rate_multiplier(affect, h)
        assert 0.05 <= lr_mult <= 2.0

    def test_memory_importance(self):
        mod = EmotionalModulation()
        affect = AffectiveState()
        imp = mod.get_memory_importance(affect)
        assert 0.0 <= imp <= 1.0

    def test_exploration_temperature(self):
        mod = EmotionalModulation()
        affect = AffectiveState()
        h = Homeostasis()
        temp = mod.get_exploration_temperature(affect, h)
        assert 0.3 <= temp <= 2.0

    def test_risk_tolerance(self):
        mod = EmotionalModulation()
        affect = AffectiveState()
        risk = mod.get_self_modification_risk_tolerance(affect)
        assert 0.0 <= risk <= 1.0
