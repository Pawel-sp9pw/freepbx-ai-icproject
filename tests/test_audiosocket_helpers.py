import unittest

from app.audiosocket import (
    apply_llm_fill_only,
    extract_phone_digits,
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

    def test_extract_phone_digits(self):
        self.assertEqual(extract_phone_digits("+48 792-032-104"), "48792032104")
        self.assertEqual(extract_phone_digits("123"), "")

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
