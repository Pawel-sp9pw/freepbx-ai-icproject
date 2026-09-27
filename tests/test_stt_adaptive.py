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

    def test_long_confident_company_skips_retry(self):
        self.assertFalse(
            stt._needs_adaptive_retry(
                "Przychodnia Zdrowie Rodzinne Katowice",
                -0.10,
                "company",
            )
        )

    def test_problem_retries_only_when_confidence_is_low(self):
        self.assertFalse(stt._needs_adaptive_retry("Nie działa recepta", -0.3, "problem"))
        self.assertTrue(stt._needs_adaptive_retry("Nie działa recepta", -0.7, "problem"))
        self.assertTrue(
            stt._needs_adaptive_retry(
                "Użytkownik nie może wystawić recepty w systemie od rana",
                -0.8,
                "problem",
            )
        )
        self.assertFalse(
            stt._needs_adaptive_retry(
                "Użytkownik nie może wystawić recepty w systemie od rana",
                -0.2,
                "problem",
            )
        )

    def test_contact_retries_only_when_invalid_or_uncertain(self):
        self.assertFalse(stt._needs_adaptive_retry("792032104", -0.1, "contact"))
        self.assertTrue(stt._needs_adaptive_retry("792032104", -0.8, "contact"))
        self.assertTrue(stt._needs_adaptive_retry("79203", -0.1, "contact"))

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
        self.assertEqual(model.calls[0]["beam_size"], 5)
        self.assertFalse(model.calls[0]["vad_filter"])
        self.assertEqual(model.calls[1]["beam_size"], 8)
        self.assertFalse(model.calls[1]["vad_filter"])

    def test_low_confidence_problem_runs_second_pass_without_prompt(self):
        model = FakeModel([
            ("Nie działa recepcja", -0.85),
            ("Nie działa recepta", -0.30),
        ])
        with patch.object(stt, "get_model", return_value=model):
            text = stt.transcribe_pcm16(
                b"\x00" * 16000,
                model_name="small",
                device="cpu",
                compute_type="int8",
                sample_rate=8000,
                initial_prompt="Opis problemu serwisowego po polsku.",
                mode="problem",
            )

        self.assertEqual(text, "Nie działa recepta")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["beam_size"], 5)
        self.assertEqual(model.calls[1]["beam_size"], 8)
        self.assertIsNone(model.calls[1]["initial_prompt"])

    def test_phone_audio_is_resampled_to_16khz(self):
        pcm = (b"\x01\x00" * 8000)
        out, rate = stt._prepare_phone_audio(pcm, 8000)
        self.assertEqual(rate, 16000)
        self.assertGreater(len(out), len(pcm))

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
