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
    def test_known_subtitle_hallucinations_are_rejected(self):
        samples = [
            "www.youtube.com www.youtube.com",
            "www.multi-moto.eu",
            "napisy stworzone przez społeczność Amara.org",
            "Transkrypcja Jan Kowalski",
            "Dziękuję za obejrzenie",
            "Dzięki za oglądanie.",
            "Dzięki za obejrzenie.",
            "Dzwoniący podaje nazwę swojej firmy po polsku.",
        ]
        for sample in samples:
            bad, reason = stt._looks_hallucinated(sample, 2.5)
            self.assertTrue(bad, sample)
            self.assertTrue(reason)

    def test_confident_short_company_skips_retry(self):
        self.assertFalse(stt._needs_adaptive_retry("Paweł", -0.49, "company"))
        self.assertFalse(stt._needs_adaptive_retry("Pizzeria Roma", -0.30, "company"))

    def test_uncertain_short_company_does_not_retry(self):
        self.assertFalse(stt._needs_adaptive_retry("Sąbeks", -0.88, "company"))
        self.assertFalse(stt._needs_adaptive_retry("Artur", -0.70, "company"))

    def test_empty_company_still_retries(self):
        self.assertTrue(stt._needs_adaptive_retry("", None, "company"))

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

    def test_contact_retries_only_when_structurally_invalid(self):
        self.assertFalse(stt._needs_adaptive_retry("792032104", -0.1, "contact"))
        self.assertFalse(stt._needs_adaptive_retry("792032104", -0.8, "contact"))
        self.assertTrue(stt._needs_adaptive_retry("79203", -0.1, "contact"))

    def test_identical_second_candidate_keeps_first_score(self):
        model = FakeModel([
            ("880227784", -0.56),
        ])
        with patch.object(stt, "get_model", return_value=model):
            result = stt.transcribe_pcm16(
                b"\x00" * 16000,
                model_name="medium",
                device="cpu",
                compute_type="int8",
                sample_rate=8000,
                initial_prompt="Numer telefonu. Cyfry od zera do dziewięciu.",
                mode="contact",
                return_metadata=True,
            )

        self.assertEqual(result["selected"], "880227784")
        self.assertEqual(result["selected_score"], -0.56)
        self.assertFalse(result["retry"])
        self.assertEqual(len(model.calls), 1)

    def test_better_second_candidate_is_selected(self):
        selected = stt._choose_candidate("Fitzseria", -0.55, "Pizzeria", -0.20)
        self.assertEqual(selected, "Pizzeria")

    def test_worse_second_candidate_is_not_selected(self):
        selected = stt._choose_candidate("Pizzeria", -0.20, "Fitzseria", -0.50)
        self.assertEqual(selected, "Pizzeria")

    def test_short_prompt_leak_skips_second_pass(self):
        model = FakeModel([
            ("Dzwoniący podaje nazwę swojej firmy po polsku. " * 5, -0.10),
        ])
        with patch.object(stt, "get_model", return_value=model):
            result = stt.transcribe_pcm16(
                b"\x00" * 9600,  # 0.6 s at 8 kHz / 16 bit
                model_name="medium",
                device="cpu",
                compute_type="int8",
                sample_rate=8000,
                initial_prompt="Dzwoniący podaje nazwę swojej firmy po polsku.",
                mode="company",
                return_metadata=True,
            )

        self.assertEqual(result["selected"], "")
        self.assertEqual(len(model.calls), 1)
        self.assertIsNone(result["pass2"])
        self.assertFalse(result["retry"])
        self.assertEqual(result["retry_reason"], "short_rejected_audio")

    def test_company_nonempty_low_confidence_skips_precision_pass(self):
        model = FakeModel([
            ("Tomek", -0.84),
        ])
        with patch.object(stt, "get_model", return_value=model):
            text = stt.transcribe_pcm16(
                b"\x00" * 16000,
                model_name="small",
                device="cpu",
                compute_type="int8",
                sample_rate=8000,
                initial_prompt="Dzwoniący podaje nazwę swojej firmy po polsku.",
                mode="company",
            )

        self.assertEqual(text, "Tomek")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["beam_size"], 5)
        self.assertFalse(model.calls[0]["vad_filter"])

    def test_company_second_pass_drops_prompt(self):
        model = FakeModel([
            ("Fitzseria", -0.55),
            ("Pizzeria", -0.18),
        ])
        with patch.object(stt, "get_model", return_value=model):
            text = stt.transcribe_pcm16(
                b"\x00" * 16000,
                model_name="medium",
                device="cpu",
                compute_type="int8",
                sample_rate=8000,
                initial_prompt="Dzwoniący podaje nazwę swojej firmy po polsku.",
                mode="company",
            )

        self.assertEqual(text, "Pizzeria")
        self.assertEqual(len(model.calls), 2)
        self.assertIsNotNone(model.calls[0]["initial_prompt"])
        self.assertIsNone(model.calls[1]["initial_prompt"])

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

    def test_rejected_prompt_leak_does_not_fall_back_to_very_weak_subtitle_hallucination(self):
        model = FakeModel([
            ("Dzwoniący opis problemu. " * 30, -0.10),
            ("Dzięki za oglądanie.", -1.031),
        ])
        with patch.object(stt, "get_model", return_value=model):
            result = stt.transcribe_pcm16(
                b"\x00" * 19200,
                model_name="medium",
                device="cpu",
                compute_type="int8",
                sample_rate=8000,
                initial_prompt="Dzwoniący opisuje problem techniczny lub usterkę po polsku.",
                mode="problem",
                return_metadata=True,
            )

        self.assertEqual(result["selected"], "")
        self.assertTrue(result["pass1"]["rejected"])
        self.assertTrue(result["pass2"]["rejected"])
        self.assertTrue(result["retry"])
        self.assertEqual(result["pass2"]["reason"], "known_whisper_hallucination")

    def test_extremely_weak_problem_candidate_is_rejected_even_if_not_known_phrase(self):
        model = FakeModel([
            ("Losowy tekst bez sensu", -1.10),
            ("Inny przypadkowy tekst", -1.02),
        ])
        with patch.object(stt, "get_model", return_value=model):
            result = stt.transcribe_pcm16(
                b"\x00" * 16000,
                model_name="medium",
                device="cpu",
                compute_type="int8",
                sample_rate=8000,
                initial_prompt="Opis problemu po polsku.",
                mode="problem",
                return_metadata=True,
            )

        self.assertEqual(result["selected"], "")
        self.assertTrue(result["pass1"]["rejected"])
        self.assertTrue(result["pass2"]["rejected"])
        self.assertIn("low_confidence", result["pass2"]["reason"])

    def test_cpu_worker_pool_splits_threads_between_workers(self):
        stt._models.clear()
        with patch.object(stt, "WhisperModel") as model_cls, \
             patch.object(stt.os, "cpu_count", return_value=8):
            stt.get_model("medium", "cpu", "int8", 2)

        kwargs = model_cls.call_args.kwargs
        self.assertEqual(kwargs["num_workers"], 2)
        self.assertEqual(kwargs["cpu_threads"], 4)

    def test_phone_audio_is_resampled_to_16khz(self):
        pcm = (b"\x01\x00" * 8000)
        out, rate = stt._prepare_phone_audio(pcm, 8000)
        self.assertEqual(rate, 16000)
        self.assertGreater(len(out), len(pcm))

    def test_metadata_mode_returns_confidence_and_passes(self):
        model = FakeModel([
            ("Alfatest", -0.20),
        ])
        with patch.object(stt, "get_model", return_value=model):
            result = stt.transcribe_pcm16(
                b"\x00" * 16000,
                model_name="medium",
                device="cpu",
                compute_type="int8",
                sample_rate=8000,
                initial_prompt="Dzwoniący podaje nazwę swojej firmy po polsku.",
                mode="normal",
                return_metadata=True,
            )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["selected"], "Alfatest")
        self.assertEqual(result["selected_score"], -0.20)
        self.assertEqual(result["pass1"]["text"], "Alfatest")
        self.assertIsNone(result["pass2"])
        self.assertFalse(result["retry"])

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
