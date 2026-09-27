import types
import unittest
from unittest.mock import patch

from app import stt


class FakeModel:
    def __init__(self, passes):
        self.passes = list(passes)
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        text, score = self.passes.pop(0)
        seg = types.SimpleNamespace(
            text=text,
            avg_logprob=score,
            start=0.0,
            end=1.0,
        )
        return iter([seg]), types.SimpleNamespace()


class AdaptiveSTTTests(unittest.TestCase):
    def test_short_company_requests_retry(self):
        self.assertTrue(stt._needs_adaptive_retry("Fitzseria", -0.10, "company"))
        self.assertTrue(stt._needs_adaptive_retry("Pizzeria Roma", -0.10, "company"))

    def test_long_company_does_not_force_retry(self):
        self.assertFalse(
            stt._needs_adaptive_retry(
                "Przychodnia Zdrowie Rodzinne Katowice",
                -0.10,
                "company",
            )
        )

    def test_problem_never_uses_company_retry_rule(self):
        self.assertFalse(stt._needs_adaptive_retry("Nie działa recepta", -0.8, "problem"))

    def test_better_second_candidate_is_selected(self):
        selected = stt._choose_candidate("Fitzseria", -0.55, "Pizzeria", -0.20)
        self.assertEqual(selected, "Pizzeria")

    def test_worse_second_candidate_is_not_selected(self):
        selected = stt._choose_candidate("Pizzeria", -0.20, "Fitzseria", -0.50)
        self.assertEqual(selected, "Pizzeria")

    def test_company_short_audio_runs_precision_pass(self):
        model = FakeModel([
            ("Fitzseria", -0.55),
            ("Pizzeria", -0.18),
        ])
        with patch.object(stt, "get_model", return_value=model):
            text = stt.transcribe_pcm16(
                b"\x00" * 16000,
                model_name="small",
                device="cpu",
                compute_type="int8",
                sample_rate=8000,
                initial_prompt="pizzeria, przychodnia",
                mode="company",
            )

        self.assertEqual(text, "Pizzeria")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["beam_size"], 3)
        self.assertTrue(model.calls[0]["vad_filter"])
        self.assertEqual(model.calls[1]["beam_size"], 5)
        self.assertFalse(model.calls[1]["vad_filter"])

    def test_confirmation_is_always_single_pass(self):
        model = FakeModel([
            ("Tak", -0.70),
        ])
        with patch.object(stt, "get_model", return_value=model):
            text = stt.transcribe_pcm16(
                b"\x00" * 16000,
                model_name="small",
                device="cpu",
                compute_type="int8",
                sample_rate=8000,
                initial_prompt="tak, nie",
                mode="confirmation",
            )

        self.assertEqual(text, "Tak")
        self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main()
