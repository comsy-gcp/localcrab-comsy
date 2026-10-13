"""Row records for large pipe tables (OPENCRAB_CHUNK_TABLE_ROWS)."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from opencrab.ontology.chunking import chunk_markdown, table_rows_enabled
from tests.test_chunking import FakeChroma, _ingest

HEADER = "| 날짜 | 대상 | 내용 |\n| --- | --- | --- |"


def _table(rows: list[str]) -> str:
    return HEADER + "\n" + "\n".join(rows)


def _rows(n: int) -> list[str]:
    return [f"| 2026-01-{i:02d} | 절 {i} | 변경 내용 {i} |" for i in range(1, n + 1)]


def _no_space(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _in_order(needle: str, haystack: str) -> bool:
    remaining = iter(haystack)
    return all(ch in remaining for ch in needle)


# ---------------------------------------------------------------------------
# Flag off: unchanged
# ---------------------------------------------------------------------------


def test_flag_off_output_matches_the_packed_markdown_form() -> None:
    # Expected bodies were produced by the code before this flag existed.
    text = "# 변경 이력\n앞 설명\n\n" + _table(_rows(5)) + "\n\n뒤 설명"
    expected = [
        f"앞 설명\n\n{HEADER}\n| 2026-01-01 | 절 1 | 변경 내용 1 |",
        f"{HEADER}\n| 2026-01-02 | 절 2 | 변경 내용 2 |",
        f"{HEADER}\n| 2026-01-03 | 절 3 | 변경 내용 3 |",
        f"{HEADER}\n| 2026-01-04 | 절 4 | 변경 내용 4 |",
        f"{HEADER}\n| 2026-01-05 | 절 5 | 변경 내용 5 |\n\n뒤 설명",
    ]

    assert [c.body for c in chunk_markdown(text, title="d", max_chars=90)] == expected
    assert [
        c.body for c in chunk_markdown(text, title="d", max_chars=90, table_rows=False)
    ] == expected


# ---------------------------------------------------------------------------
# One record per row
# ---------------------------------------------------------------------------


def test_large_table_becomes_one_record_per_row() -> None:
    rows = _rows(20)
    text = "# 변경 이력\n" + _table(rows)

    chunks = chunk_markdown(text, title="d", max_chars=200, table_rows=True)

    assert len(chunks) == 20
    assert chunks[0].body == "날짜: 2026-01-01\n대상: 절 1\n내용: 변경 내용 1"
    assert chunks[19].body == "날짜: 2026-01-20\n대상: 절 20\n내용: 변경 내용 20"
    assert all(c.heading_path == ("변경 이력",) for c in chunks)
    assert [c.index for c in chunks] == list(range(20))


def test_table_that_fits_the_section_is_kept_as_markdown() -> None:
    text = "# 변경 이력\n" + _table(_rows(2))

    chunks = chunk_markdown(text, title="d", max_chars=200, table_rows=True)

    assert len(chunks) == 1
    assert chunks[0].body.startswith(HEADER)


def test_small_table_in_long_section_is_packed_as_markdown() -> None:
    para = "설명 문단 " + "가" * 150
    text = "# 절\n" + para + "\n\n" + _table(_rows(1)) + "\n\n" + para

    chunks = chunk_markdown(text, title="d", max_chars=200, table_rows=True)

    assert any(HEADER in c.body for c in chunks)
    assert not any(c.body.startswith("날짜:") for c in chunks)


def test_text_around_large_table_is_not_merged_into_row_records() -> None:
    text = "# 변경 이력\n앞 설명\n\n" + _table(_rows(10)) + "\n\n뒤 설명"

    chunks = chunk_markdown(text, title="d", max_chars=200, table_rows=True)

    bodies = [c.body for c in chunks]
    assert bodies[0] == "앞 설명"
    assert bodies[-1] == "뒤 설명"
    assert bodies[1:-1] == [
        f"날짜: 2026-01-{i:02d}\n대상: 절 {i}\n내용: 변경 내용 {i}" for i in range(1, 11)
    ]


def test_empty_cells_are_skipped_and_unnamed_columns_get_numbers() -> None:
    header = "| 날짜 |  | 내용 |\n| --- | --- | --- |"
    rows = [f"| 2026-02-{i:02d} | 비고 {i} |  | 넘침 {i} |" for i in range(1, 11)]
    text = "# 표\n" + header + "\n" + "\n".join(rows)

    chunks = chunk_markdown(text, title="d", max_chars=150, table_rows=True)

    assert chunks[0].body == "날짜: 2026-02-01\n열 2: 비고 1\n열 4: 넘침 1"


def test_escaped_pipe_stays_inside_its_cell() -> None:
    rows = [f"| 2026-03-{i:02d} | a \\| b {i} | 내용 {i} |" for i in range(1, 11)]
    text = "# 표\n" + _table(rows)

    chunks = chunk_markdown(text, title="d", max_chars=150, table_rows=True)

    assert chunks[0].body == "날짜: 2026-03-01\n대상: a \\| b 1\n내용: 내용 1"


def test_trailing_escaped_pipe_is_part_of_the_last_cell() -> None:
    rows = [f"| 2026-06-{i:02d} | 대상 {i} | 끝 \\|" for i in range(1, 11)]
    text = "# 표\n" + _table(rows)

    chunks = chunk_markdown(text, title="d", max_chars=150, table_rows=True)

    assert chunks[0].body == "날짜: 2026-06-01\n대상: 대상 1\n내용: 끝 \\|"


def test_pipes_inside_inline_code_without_escape_still_split_cells() -> None:
    # GFM requires escaping pipes even inside code spans, so this is three cells.
    rows = [f"| 2026-04-{i:02d} | `x | y` |" for i in range(1, 11)]
    text = "# 표\n" + _table(rows)

    chunks = chunk_markdown(text, title="d", max_chars=150, table_rows=True)

    assert chunks[0].body == "날짜: 2026-04-01\n대상: `x\n내용: y`"


def test_rows_without_outer_pipes_are_parsed() -> None:
    header = "| 날짜 | 내용 |\n|---|---|"
    rows = [f"| 2026-05-{i:02d} | 내용 {i}" for i in range(1, 11)]
    text = "# 표\n" + header + "\n" + "\n".join(rows)

    chunks = chunk_markdown(text, title="d", max_chars=100, table_rows=True)

    assert chunks[0].body == "날짜: 2026-05-01\n내용: 내용 1"


# ---------------------------------------------------------------------------
# Rows longer than the limit
# ---------------------------------------------------------------------------


def _long_row_table(reason: str, change: str = "짧은 변경") -> str:
    header = "| 적용일 | 변경 | 이유 |\n| --- | --- | --- |"
    rows = [f"| 2026-09-{i:02d} | 다른 변경 {i} | 다른 이유 {i} |" for i in range(1, 6)]
    rows.insert(2, f"| 2026-09-29 | {change} | {reason} |")
    return "# 변경 이력\n" + header + "\n" + "\n".join(rows)


def test_long_row_pieces_repeat_the_first_column() -> None:
    reason = " ".join(f"사유{i}" for i in range(120))
    text = _long_row_table(reason)

    chunks = chunk_markdown(text, title="d", max_chars=200, table_rows=True)

    pieces = [c.body for c in chunks if "2026-09-29" in c.body]
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    assert all(p.startswith("적용일: 2026-09-29\n") for p in pieces)
    assert pieces[0] == "적용일: 2026-09-29\n변경: 짧은 변경"
    assert all(p.startswith("적용일: 2026-09-29\n이유: ") for p in pieces[1:])
    tails = "".join(p.split("이유: ", 1)[1] for p in pieces[1:])
    assert _no_space(tails) == _no_space(reason)


def test_long_row_keeps_short_cells_whole_and_in_order() -> None:
    change = "변" * 150
    reason = "이" * 150
    text = _long_row_table(reason, change=change)

    chunks = chunk_markdown(text, title="d", max_chars=200, table_rows=True)

    pieces = [c.body for c in chunks if "2026-09-29" in c.body]
    assert pieces == [
        f"적용일: 2026-09-29\n변경: {change}",
        f"적용일: 2026-09-29\n이유: {reason}",
    ]


def test_huge_first_cell_is_not_repeated() -> None:
    key = "키" * 120
    rows = [f"| {key}{i} | 값 {i} |" for i in range(3)]
    text = "# 표\n| 이름 | 값 |\n| --- | --- |\n" + "\n".join(rows)

    chunks = chunk_markdown(text, title="d", max_chars=100, table_rows=True)

    assert all(len(c.body) <= 100 for c in chunks)
    assert sum(c.body.count("값: ") for c in chunks) == 3
    assert all(c.body.startswith("이름: ") for c in chunks if "값: " not in c.body)
    assert _no_space("".join(c.body for c in chunks)).count("키") == 360


@pytest.mark.parametrize("max_chars", [5, 12, 30, 60])
def test_tiny_limits_never_exceed_max_or_drop_cell_text(max_chars: int) -> None:
    # ASCII column names and dates, Hangul values: every Hangul character in the
    # output must come from a cell value, and none may be missing.
    values = [(f"긴변경설명{i}" * 3, f"이유{i}가나다") for i in range(1, 4)]
    rows = [f"| 2026-09-0{i} | {v} | {r} |" for i, (v, r) in enumerate(values, 1)]
    text = "# T\n| Date | Change | Why |\n| --- | --- | --- |\n" + "\n".join(rows)

    chunks = chunk_markdown(text, title="d", max_chars=max_chars, table_rows=True)

    assert all(0 < len(c.body) <= max_chars for c in chunks)
    hangul = Counter(ch for c in chunks for ch in c.body if "가" <= ch <= "힣")
    assert hangul == Counter(ch for v, r in values for ch in v + r if "가" <= ch <= "힣")


def test_table_with_only_empty_cells_keeps_its_header() -> None:
    text = "# 표\n| 날짜 | 대상 |\n| --- | --- |\n|  | |\n|  |  |"

    chunks = chunk_markdown(text, title="d", max_chars=20, table_rows=True)

    assert "날짜" in "".join(c.body for c in chunks)
    assert all(len(c.body) <= 20 for c in chunks)


def test_empty_first_cell_repeats_the_first_non_empty_cell() -> None:
    long_value = " ".join(["내용"] * 60)
    rows = [f"| | 키{i} | {long_value} |" for i in range(3)]
    text = "# 표\n| 날짜 | 대상 | 내용 |\n| --- | --- | --- |\n" + "\n".join(rows)

    chunks = chunk_markdown(text, title="d", max_chars=100, table_rows=True)

    assert all(c.body.startswith(("대상: 키0\n", "대상: 키1\n", "대상: 키2\n")) for c in chunks)


def test_split_cell_pieces_have_a_single_space_after_the_label() -> None:
    reason = " ".join(f"사유{i}" for i in range(120))
    text = _long_row_table(reason)

    chunks = chunk_markdown(text, title="d", max_chars=200, table_rows=True)

    assert not any(":  " in c.body for c in chunks)


# ---------------------------------------------------------------------------
# Environment flag and ingest wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"), [("1", True), ("on", True), ("", False), ("0", False)]
)
def test_table_rows_env_parsing(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("OPENCRAB_CHUNK_TABLE_ROWS", raw)

    assert table_rows_enabled() is expected


def test_ingest_uses_row_records_when_flag_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCRAB_CHUNKING", "1")
    monkeypatch.setenv("OPENCRAB_CHUNK_MAX_CHARS", "200")
    monkeypatch.setenv("OPENCRAB_CHUNK_TABLE_ROWS", "1")
    chroma = FakeChroma()

    result = _ingest(chroma, "# 변경 이력\n" + _table(_rows(20)), "log-1", {"title": "변경 이력"})

    assert result["chunks"] == 20
    text, meta = chroma.records["log-1#3"]
    assert (
        text == "문서: 변경 이력\n절: 변경 이력\n\n날짜: 2026-01-04\n대상: 절 4\n내용: 변경 내용 4"
    )
    assert meta["chunk_index"] == 3


def test_ingest_keeps_markdown_rows_when_flag_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCRAB_CHUNKING", "1")
    monkeypatch.setenv("OPENCRAB_CHUNK_MAX_CHARS", "200")
    monkeypatch.delenv("OPENCRAB_CHUNK_TABLE_ROWS", raising=False)
    chroma = FakeChroma()

    _ingest(chroma, "# 변경 이력\n" + _table(_rows(20)), "log-1")

    assert all(HEADER in t for t, _ in chroma.records.values())
