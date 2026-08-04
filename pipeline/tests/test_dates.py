from __future__ import annotations

import unittest

from pipeline.dates import DateConfig, normalize_date, uniform_bin_weights


class DateNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DateConfig(bin_size=10, circa_years=5, open_range_years=25)

    def test_exact_date_has_unit_weight(self) -> None:
        date = normalize_date({"date_begin": "1885", "date_end": "1885"}, self.config)
        self.assertEqual((date.start, date.end, date.qualifier), (1885, 1885, "exact"))
        self.assertEqual(uniform_bin_weights(date, 10), {1880: 1.0})

    def test_range_is_distributed_by_inclusive_year_overlap(self) -> None:
        date = normalize_date({"date_begin": "1880", "date_end": "1890"}, self.config)
        weights = uniform_bin_weights(date, 10)
        self.assertAlmostEqual(weights[1880], 10 / 11)
        self.assertAlmostEqual(weights[1890], 1 / 11)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_circa_expands_exact_date(self) -> None:
        date = normalize_date(
            {"date_begin": "1880", "date_end": "1880", "date_qualifier": "circa"},
            self.config,
        )
        self.assertEqual((date.start, date.end, date.parse_method), (1875, 1885, "source_circa"))
        self.assertAlmostEqual(sum(uniform_bin_weights(date, 10).values()), 1.0)

    def test_display_circa_and_open_ranges(self) -> None:
        circa = normalize_date({"date_display": "ca. 1880"}, self.config)
        before = normalize_date({"date_display": "before 1900"}, self.config)
        after = normalize_date({"date_display": "after 1900"}, self.config)
        self.assertEqual((circa.start, circa.end), (1875, 1885))
        self.assertEqual((before.start, before.end), (1875, 1899))
        self.assertEqual((after.start, after.end), (1901, 1925))

    def test_unknown_date_has_no_weights(self) -> None:
        date = normalize_date({"date_display": "Undated"}, self.config)
        self.assertFalse(date.dated)
        self.assertEqual(date.qualifier, "unknown")
        self.assertEqual(uniform_bin_weights(date, 10), {})

    def test_bce_flags_create_ordered_negative_range(self) -> None:
        date = normalize_date(
            {
                "date_begin": "2600",
                "date_end": "2000",
                "date_begin_bce": "true",
                "date_end_bce": "true",
            },
            self.config,
        )
        self.assertEqual((date.start, date.end), (-2600, -2000))
        self.assertAlmostEqual(sum(uniform_bin_weights(date, 10).values()), 1.0)

    def test_display_bce_range_inherits_end_era(self) -> None:
        date = normalize_date({"date_display": "300-200 BCE"}, self.config)
        self.assertEqual((date.start, date.end), (-300, -200))
        self.assertAlmostEqual(sum(uniform_bin_weights(date, 10).values()), 1.0)

        start_era = normalize_date({"date_display": "300 BCE-200"}, self.config)
        self.assertEqual((start_era.start, start_era.end), (-300, -200))

    def test_cross_era_range_skips_year_zero_and_sums_to_one(self) -> None:
        date = normalize_date(
            {
                "date_begin": "2",
                "date_end": "2",
                "date_begin_bce": "true",
                "date_end_bce": "false",
            },
            self.config,
        )
        self.assertEqual((date.start, date.end), (-2, 2))
        self.assertAlmostEqual(sum(uniform_bin_weights(date, 10).values()), 1.0)


if __name__ == "__main__":
    unittest.main()
