"""Cierra: a colour-grading language. Grades are small checked programs that compile to any target."""

__version__ = "0.1.0"
LANGUAGE_VERSION = "0.1"

from .errors import CierraError, CheckError, ParseError  # noqa: E402,F401
from .parser import parse  # noqa: E402,F401
from .checker import check, load  # noqa: E402,F401
from .format import format_program  # noqa: E402,F401
