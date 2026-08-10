"""Anki Cozmo brand tokens for the CLI.

Real Cozmo (Anki / Digital Dream Labs):
  - white plastic body + black OLED face (128×32, interlaced)
  - pupil-less cyan/blue LED eyes
  - emotions via eye *shape*, not color
"""

from __future__ import annotations

# OLED eye glow (matches the robot’s signature blue)
CYAN = "cyan"
CYAN_BOLD = "bold cyan"
CYAN_DIM = "dim cyan"

# Headphones / accent purple (mascot reference)
HEADPHONE = "bold #6b5ce0"
HEADPHONE_DIM = "#6b5ce0"

# Body shell (charcoal bezel)
SHELL = "bright_white"
SHELL_DIM = "dim white"

# Dim chrome for secondary status
MUTED = "dim"

# questionary Style values (prompt-toolkit color names)
QUESTIONARY = (
    ("qmark", "fg:cyan bold"),
    ("question", "fg:cyan bold"),
    ("answer", "fg:cyan bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:cyan"),
    ("instruction", "fg:#6b7280"),
)
