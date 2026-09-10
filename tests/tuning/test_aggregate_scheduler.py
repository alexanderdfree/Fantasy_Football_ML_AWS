import pytest

from src.tuning.aggregate_scheduler import _md_table


@pytest.mark.unit
def test_scheduler_markdown_table_has_matching_column_counts():
    table = _md_table(
        [
            {
                "position": "QB",
                "production_type": "onecycle",
                "aggregated": {},
                "verdict": {},
                "sentinel_ok": True,
            }
        ]
    )
    header, delimiter, row = table.splitlines()[:3]

    def columns(line):
        return len(line.strip("|").split("|"))

    assert columns(header) == columns(delimiter) == columns(row) == 8
