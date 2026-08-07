"""Versioned date normalization and uniform date-to-bin weighting."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Mapping


DATE_RULES_VERSION = "artifact-bootstrap-dates/v1"
UNKNOWN_DATE_TERMS = {"", "unknown", "undated", "n.d.", "n.d", "none", "null", "nan"}


@dataclass(frozen=True)
class DateConfig:
    """Rules whose values must be recorded in a corpus manifest."""

    bin_size: int = 10
    circa_years: int = 5
    open_range_years: int = 25
    min_year: int | None = None
    max_year: int | None = None

    def validate(self) -> None:
        if self.bin_size <= 0:
            raise ValueError("bin_size must be positive")
        if self.circa_years < 0:
            raise ValueError("circa_years must be non-negative")
        if self.open_range_years <= 0:
            raise ValueError("open_range_years must be positive")
        if self.min_year == 0 or self.max_year == 0:
            raise ValueError("year zero is not valid in historical year numbering")
        if self.min_year is not None and self.max_year is not None:
            if self.min_year > self.max_year:
                raise ValueError("min_year must not exceed max_year")


@dataclass(frozen=True)
class NormalizedDate:
    display: str
    start: int | None
    end: int | None
    qualifier: str
    parse_method: str

    @property
    def dated(self) -> bool:
        return self.start is not None and self.end is not None


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_bool(value: object) -> bool:
    return _text(value).lower() in {"1", "true", "t", "yes", "y"}


def parse_year(value: object) -> int | None:
    text = _text(value).replace(",", "")
    if text.lower() in UNKNOWN_DATE_TERMS:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if not number.is_integer():
        raise ValueError(f"date year must be an integer, got {value!r}")
    year = int(number)
    if year == 0:
        raise ValueError("year zero is not valid in historical year numbering")
    return year


def _historical_year(year: int | None, is_bce: bool) -> int | None:
    if year is None:
        return None
    return -abs(year) if is_bce else year


def _canonical_qualifier(value: object) -> str:
    text = _text(value).lower()
    if text in {"c.", "ca", "ca.", "circa", "about", "approximately"}:
        return "circa"
    if text.startswith("before") or text in {"pre", "earlier than"}:
        return "before"
    if text.startswith("after") or text in {"post", "later than"}:
        return "after"
    if text in UNKNOWN_DATE_TERMS:
        return ""
    return text


def _format_year(year: int) -> str:
    return f"{abs(year)} BCE" if year < 0 else str(year)


def _default_display(start: int | None, end: int | None, qualifier: str) -> str:
    if start is None or end is None:
        return "Unknown"
    if qualifier == "before":
        return f"before {_format_year(end + 1 if end != -1 else 1)}"
    if qualifier == "after":
        return f"after {_format_year(start - 1 if start != 1 else -1)}"
    if start == end:
        prefix = "circa " if qualifier == "circa" else ""
        return f"{prefix}{_format_year(start)}"
    return f"{_format_year(start)}–{_format_year(end)}"


_YEAR_TOKEN = r"(?P<year>\d{1,5})(?:\s*(?P<era>BCE|BC|CE|AD))?"
_SINGLE_RE = re.compile(rf"^\s*{_YEAR_TOKEN}\s*$", re.IGNORECASE)
_RANGE_RE = re.compile(
    rf"^\s*(?P<start>\d{{1,5}})(?:\s*(?P<start_era>BCE|BC|CE|AD))?"
    rf"\s*(?:-|–|—|to)\s*"
    rf"(?P<end>\d{{1,5}})(?:\s*(?P<end_era>BCE|BC|CE|AD))?\s*$",
    re.IGNORECASE,
)


def _apply_era(year: int, era: str | None, inherited_era: str | None = None) -> int:
    if year == 0:
        raise ValueError("year zero is not valid in historical year numbering")
    selected = (era or inherited_era or "").upper()
    return -abs(year) if selected in {"BCE", "BC"} else year


def _parse_display(display: str, config: DateConfig) -> NormalizedDate | None:
    stripped = display.strip()
    lowered = stripped.lower()
    if lowered in UNKNOWN_DATE_TERMS:
        return NormalizedDate(display or "Unknown", None, None, "unknown", "unknown")

    qualifier = ""
    body = stripped
    for pattern, canonical in (
        (r"^(?:circa|ca\.?|c\.?|about|approximately)\s+", "circa"),
        (r"^before\s+", "before"),
        (r"^after\s+", "after"),
    ):
        match = re.match(pattern, body, flags=re.IGNORECASE)
        if match:
            qualifier = canonical
            body = body[match.end() :]
            break

    range_match = _RANGE_RE.match(body)
    if range_match:
        end_era = range_match.group("end_era")
        start_era = range_match.group("start_era")
        start = _apply_era(
            int(range_match.group("start")), start_era, end_era
        )
        end = _apply_era(int(range_match.group("end")), end_era, start_era)
        if start > end:
            start, end = end, start
        return NormalizedDate(stripped, start, end, qualifier or "range", "display_range")

    single_match = _SINGLE_RE.match(body)
    if not single_match:
        return None
    year = _apply_era(int(single_match.group("year")), single_match.group("era"))
    if qualifier == "circa":
        start, end = shift_year(year, -config.circa_years), shift_year(year, config.circa_years)
        return NormalizedDate(stripped, start, end, qualifier, "display_circa")
    if qualifier == "before":
        return NormalizedDate(
            stripped,
            shift_year(year, -config.open_range_years),
            shift_year(year, -1),
            qualifier,
            "display_before",
        )
    if qualifier == "after":
        return NormalizedDate(
            stripped,
            shift_year(year, 1),
            shift_year(year, config.open_range_years),
            qualifier,
            "display_after",
        )
    return NormalizedDate(stripped, year, year, "exact", "display_exact")


def shift_year(year: int, offset: int) -> int:
    """Shift a historical year while skipping the nonexistent year zero."""

    shifted = year + offset
    if year < 0 <= shifted:
        shifted += 1
    elif year > 0 >= shifted:
        shifted -= 1
    return shifted


def normalize_date(row: Mapping[str, object], config: DateConfig) -> NormalizedDate:
    """Normalize canonical or ArtiFact date fields without inventing unknown dates."""

    config.validate()
    display = _text(row.get("date_display"))
    qualifier = _canonical_qualifier(row.get("date_qualifier"))
    start_raw = row.get("date_start", row.get("date_begin"))
    end_raw = row.get("date_end")
    start = parse_year(start_raw)
    end = parse_year(end_raw)
    start = _historical_year(start, parse_bool(row.get("date_start_bce", row.get("date_begin_bce"))))
    end = _historical_year(end, parse_bool(row.get("date_end_bce")))

    if start is None and end is None:
        parsed = _parse_display(display, config) if display else None
        return parsed or NormalizedDate(display or "Unknown", None, None, "unknown", "unknown")
    if start is None:
        start = end
    if end is None:
        end = start
    assert start is not None and end is not None
    if start > end:
        start, end = end, start

    method = "source_exact" if start == end else "source_range"
    effective_qualifier = qualifier or ("exact" if start == end else "range")
    if effective_qualifier == "circa" and start == end:
        center = start
        start = shift_year(center, -config.circa_years)
        end = shift_year(center, config.circa_years)
        method = "source_circa"
    elif effective_qualifier == "before":
        boundary = end
        start = shift_year(boundary, -config.open_range_years)
        end = shift_year(boundary, -1)
        method = "source_before"
    elif effective_qualifier == "after":
        boundary = start
        start = shift_year(boundary, 1)
        end = shift_year(boundary, config.open_range_years)
        method = "source_after"

    return NormalizedDate(
        display or _default_display(start, end, effective_qualifier),
        start,
        end,
        effective_qualifier,
        method,
    )


def iter_historical_years(start: int, end: int) -> Iterable[int]:
    for year in range(start, end + 1):
        if year != 0:
            yield year


def bin_start_for_year(year: int, bin_size: int) -> int:
    if year == 0:
        raise ValueError("year zero is not valid in historical year numbering")
    return math.floor(year / bin_size) * bin_size


def bin_end_for_start(start: int, bin_size: int) -> int:
    end = start + bin_size - 1
    if start == 0:
        return max(1, end)
    return end


def bin_label(start: int, end: int) -> str:
    years = [year for year in range(start, end + 1) if year != 0]
    if not years:
        return ""
    first, last = years[0], years[-1]
    if first < 0 and last < 0:
        return f"{abs(first)}–{abs(last)} BCE"
    if first > 0 and last > 0:
        return f"{first}–{last} CE"
    return f"{_format_year(first)}–{_format_year(last)}"


def make_bins(dates: Iterable[NormalizedDate], config: DateConfig) -> list[tuple[int, int]]:
    dated = [date for date in dates if date.dated]
    if not dated and (config.min_year is None or config.max_year is None):
        return []
    dated_minimum = min((date.start for date in dated if date.start is not None), default=None)
    dated_maximum = max((date.end for date in dated if date.end is not None), default=None)
    minimum_candidates = [year for year in (config.min_year, dated_minimum) if year is not None]
    maximum_candidates = [year for year in (config.max_year, dated_maximum) if year is not None]
    minimum = min(minimum_candidates)
    maximum = max(maximum_candidates)
    assert minimum is not None and maximum is not None
    starts = sorted({bin_start_for_year(year, config.bin_size) for year in iter_historical_years(minimum, maximum)})
    return [(start, bin_end_for_start(start, config.bin_size)) for start in starts]


def uniform_bin_weights(date: NormalizedDate, bin_size: int) -> dict[int, float]:
    if not date.dated:
        return {}
    assert date.start is not None and date.end is not None
    years = list(iter_historical_years(date.start, date.end))
    if not years:
        return {}
    counts: dict[int, int] = {}
    for year in years:
        start = bin_start_for_year(year, bin_size)
        counts[start] = counts.get(start, 0) + 1
    denominator = len(years)
    return {start: count / denominator for start, count in sorted(counts.items())}
