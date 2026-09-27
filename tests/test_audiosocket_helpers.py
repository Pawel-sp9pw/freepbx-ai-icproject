import unittest

from app.audiosocket import (
    apply_llm_fill_only,
    extract_phone_digits,
    looks_like_invalid_company_name,
    match_customer,
    matches_confirmation_phrase,
    parse_customer_directory,
)


class AudioSocketRegressionTests(unittest.TestCase):
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
        customer, score = match_customer("Atur", "", directory)
        self.assertIsNone(customer)
        self.assertGreater(score, 0.0)

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
