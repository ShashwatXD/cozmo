"""Activity spinner helpers."""

from cozmo.cli.activity import tool_label


def test_tool_labels() -> None:
    assert tool_label("search_repo") == "finding files"
    assert tool_label("read_file") == "reading"
    assert "foo bar" == tool_label("foo_bar")
