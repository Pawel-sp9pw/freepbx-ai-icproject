import unittest

from app.audiosocket import (
    apply_llm_fill_only,
    clean_company_display_name,
    extract_phone_digits,
    looks_like_invalid_company_name,
    looks_like_ticket_cancellation,
    match_customer,
    matches_confirmation_phrase,
    parse_customer_directory,
)


class AudioSocketRegressionTests(unittest.TestCase):

    def test_phone_parser_accepts_observed_ordinal_variants(self):
        self.assertEqual(
            extract_phone_digits(
                "Pięćset pięć, jedenasty, osiemdziesiąt cztery, osiąnasty.",
                "pl",
            ),
            "505118418",
        )
        self.assertEqual(
            extract_phone_digits(
                "Osiemset dwudziesta sześć pięćdziesiąt dziewięć dwudziesta osiem dziewięćdziesiąt pięć.",
                "pl",
            ),
            "826592895",
        )


    def test_company_noise_and_repetition_are_rejected(self):
        samples = [
            "Nie, nie, nie, nie, nie, nie, nie, nie, nie, nie.",
            "Do zobaczenia.",
            "No, no, no, to jest...",
            "Dziękuję.",
            "tak",
        ]
        for sample in samples:
            self.assertTrue(
                looks_like_invalid_company_name(sample),
                sample,
            )

    def test_normal_company_names_are_not_rejected_as_noise(self):
        for sample in ("Przychodnia Testowa Beta", "Apteka Testowa Pod Dębem", "No Problem IT"):
            self.assertFalse(
                looks_like_invalid_company_name(sample),
                sample,
            )

    def test_parse_customer_directory(self):
        directory = parse_customer_directory(
            "Paweł | 792 032 104\nPizzeria Roma | +48 600-100-200\n"
        )
        self.assertEqual(directory[0], {"name": "Paweł", "phone": "792032104"})
        self.assertEqual(directory[1], {"name": "Pizzeria Roma", "phone": "48600100200"})

    def test_exact_phone_matches_customer(self):
        directory = [{"name": "Paweł", "phone": "792032104"}]
        customer, score = match_customer("", "792032104", directory)
        self.assertEqual(customer["name"], "Paweł")
        self.assertEqual(score, 1.0)

    def test_country_code_suffix_match_is_allowed(self):
        directory = [{"name": "Paweł", "phone": "792032104"}]
        customer, score = match_customer("", "48792032104", directory)
        self.assertEqual(customer["name"], "Paweł")
        self.assertEqual(score, 1.0)

    def test_short_directory_phone_does_not_suffix_match(self):
        directory = [{"name": "Wrong", "phone": "104"}]
        customer, score = match_customer("", "792032104", directory)
        self.assertIsNone(customer)
        self.assertEqual(score, 0.0)

    def test_unknown_phone_does_not_match(self):
        directory = [{"name": "Paweł", "phone": "792032104"}]
        customer, _ = match_customer("", "600111222", directory)
        self.assertIsNone(customer)

    def test_fuzzy_customer_match_repairs_common_short_name_errors(self):
        directory = [
            {"name": "Artur", "phone": "604943274"},
            {"name": "Marcin", "phone": "608411319"},
            {"name": "Tomek", "phone": "790205140"},
        ]

        customer, score = match_customer("Marci", "", directory)
        self.assertEqual(customer["name"], "Marcin")
        self.assertGreaterEqual(score, 0.62)

        customer, score = match_customer("Atul", "", directory)
        self.assertEqual(customer["name"], "Artur")
        self.assertGreaterEqual(score, 0.62)

        customer, score = match_customer("Firma Tomek", "", directory)
        self.assertEqual(customer["name"], "Tomek")
        self.assertEqual(score, 1.0)

        customer, score = match_customer("A fataszt.", "", [
            {"name": "Alfatest", "phone": "880227784"},
            {"name": "Tomek", "phone": "790205140"},
        ])
        self.assertEqual(customer["name"], "Alfatest")
        self.assertGreaterEqual(score, 0.72)

    def test_fuzzy_customer_match_does_not_guess_ambiguous_short_name(self):
        directory = [
            {"name": "Artur", "phone": "604943274"},
            {"name": "Artus", "phone": "600000001"},
        ]
        customer, score = match_customer("Artu", "", directory)
        self.assertIsNone(customer)
        self.assertGreater(score, 0.0)

    def test_clean_company_display_name_removes_only_conversational_prefixes(self):
        self.assertEqual(clean_company_display_name("Firma Alfatest"), "Alfatest")
        self.assertEqual(clean_company_display_name("Dzień dobry, tu firma Alfatest"), "Alfatest")
        self.assertEqual(clean_company_display_name("Spółka Alfa Med"), "Alfa Med")
        self.assertEqual(clean_company_display_name("Alfa Firma Serwis"), "Alfa Firma Serwis")

    def test_ticket_cancellation_tolerates_observed_stt_distortions(self):
        self.assertTrue(looks_like_ticket_cancellation(
            "Właściwie to już zaczęło działać. Proszę unie zakładać zgłoszenia."
        ))
        self.assertTrue(looks_like_ticket_cancellation(
            "Proszę omułować zgłoszenie. Już nie jest potrzebne."
        ))
        self.assertTrue(looks_like_ticket_cancellation(
            "Wie pan co? Prozygnuje ze zgłuszenia, sam to sprawdzi."
        ))
        self.assertFalse(looks_like_ticket_cancellation(
            "Zgłoszenie jest potrzebne, proszę je zapisać."
        ))

    def test_extract_phone_digits(self):
        self.assertEqual(extract_phone_digits("792-032-104"), "792032104")
        self.assertEqual(extract_phone_digits("+48 792-032-104"), "792032104")
        self.assertEqual(extract_phone_digits("0048 792 032 104"), "792032104")
        self.assertEqual(
            extract_phone_digits(
                "Osiemset osiemdziesiąt dwadziesta dwa siedemdziesiąt siedem osiemdziesiąt cztery."
            ),
            "880227784",
        )
        self.assertEqual(extract_phone_digits("siedem dziewięć dwa zero trzy dwa jeden zero cztery"), "792032104")
        self.assertEqual(
            extract_phone_digits("Mój numer to siedemset trzydzieści pięć, siedemdziesiąt trzy, siedemdziesiąt sześć, dwadzieścia osiem"),
            "735737628",
        )
        self.assertEqual(extract_phone_digits("Mój numer to 7357, czy 7628."), "735737628")
        self.assertEqual(extract_phone_digits("Mój numer to 7 3 5 7 czy 7 6 2 8."), "735737628")
        self.assertEqual(
            extract_phone_digits("Osiemset osiemdziesiąt trzy, sześćsty trzy, sto trzydzieści dziewięć."),
            "883603139",
        )
        self.assertEqual(extract_phone_digits("599-3131-2120"), "")
        self.assertEqual(extract_phone_digits("123"), "")
        self.assertEqual(extract_phone_digits("+44 20 7946 0958", "pl"), "")
        self.assertEqual(extract_phone_digits("+44 20 7946 0958", "international"), "442079460958")

    def test_rejects_whisper_company_hallucinations(self):
        self.assertTrue(looks_like_invalid_company_name("www.youtube.com www.youtube.com"))
        self.assertTrue(looks_like_invalid_company_name("napisy stworzone przez społeczność Amara.org"))
        self.assertTrue(looks_like_invalid_company_name("www.multi-moto.eu"))
        self.assertFalse(looks_like_invalid_company_name("Firma Alfatest"))
        self.assertFalse(looks_like_invalid_company_name("Pizzeria Roma"))

    def test_confirmation_requires_exact_phrase(self):
        yes = ("tak", "zgadza się", "potwierdzam")
        self.assertTrue(matches_confirmation_phrase("Tak.", yes))
        self.assertTrue(matches_confirmation_phrase("zgadza się", yes))
        self.assertFalse(matches_confirmation_phrase("tak chyba", yes))
        self.assertFalse(matches_confirmation_phrase("nie tak", yes))

    def test_llm_fill_only_does_not_overwrite_existing_values(self):
        ticket = {
            "company": "Paweł",
            "contact": "792032104",
            "description": "",
        }
        update = {
            "company": "Administrator",
            "contact": "999999999",
            "description": "Nie działa system",
            "priority": "high",
            "caller": "111111111",
        }
        result = apply_llm_fill_only(ticket, update)
        self.assertEqual(result["company"], "Paweł")
        self.assertEqual(result["contact"], "792032104")
        self.assertEqual(result["description"], "Nie działa system")
        self.assertNotIn("priority", result)
        self.assertNotIn("caller", result)

    def test_llm_fill_only_cannot_erase_values(self):
        ticket = {"company": "Paweł", "description": "Problem"}
        result = apply_llm_fill_only(ticket, {"company": "", "description": None})
        self.assertEqual(result["company"], "Paweł")
        self.assertEqual(result["description"], "Problem")


if __name__ == "__main__":
    unittest.main()
