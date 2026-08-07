from __future__ import annotations

import unittest

from mnemosyne_search.parsing import MAX_QUERY_LENGTH, QuerySyntaxError, normalize_term, parse_query


class QueryParsingTests(unittest.TestCase):
    def test_parses_quoted_commas_and_preserves_labels(self) -> None:
        terms = parse_query('horse, "still life, fruit", train')
        self.assertEqual([term.label for term in terms], ["horse", "still life, fruit", "train"])
        self.assertEqual([term.normalized for term in terms], ["horse", "still life, fruit", "train"])

    def test_deduplicates_normalized_terms_in_first_seen_order(self) -> None:
        terms = parse_query(" Horse , horse, SHIP ")
        self.assertEqual([term.label for term in terms], ["Horse", "SHIP"])
        self.assertEqual(terms[0].id, parse_query("horse")[0].id)

    def test_normalizes_unicode_and_whitespace(self) -> None:
        self.assertEqual(normalize_term("  ＨＯＲＳＥ\n study "), "horse study")

    def test_matches_frontend_unicode_lowercase_semantics(self) -> None:
        terms = parse_query("Straße, STRASSE")
        self.assertEqual([term.normalized for term in terms], ["straße", "strasse"])

    def test_supports_doubled_quote_escape(self) -> None:
        terms = parse_query('"portrait of ""Ada"", seated", horse')
        self.assertEqual(terms[0].label, 'portrait of "Ada", seated')

    def test_rejects_invalid_queries(self) -> None:
        for raw in ("", "horse,", ",horse", '"horse', "a,b,c,d,e,f", "x" * (MAX_QUERY_LENGTH + 1)):
            with self.subTest(raw=raw), self.assertRaises(QuerySyntaxError):
                parse_query(raw)


if __name__ == "__main__":
    unittest.main()
