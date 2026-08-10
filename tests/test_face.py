"""Cozmo splash rendering."""

from cozmo.cli.face import render_face_plain
from cozmo.cli.mascot_ascii import MASCOT_ASCII


def test_mascot_ascii_shape() -> None:
    assert len(MASCOT_ASCII) == 30
    assert all(len(row) == len(MASCOT_ASCII[0]) for row in MASCOT_ASCII)
    assert "@" in "".join(MASCOT_ASCII)


def test_splash_includes_mascot_and_wordmark() -> None:
    face = render_face_plain()
    assert "@@@@" in face
    assert "██████╗" in face
