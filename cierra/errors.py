"""Errors. Every message says where (line, node, parameter) and what to change, so an LLM can fix its program."""


class CierraError(ValueError):
    pass


class ParseError(CierraError):
    def __init__(self, msg, line=None, col=None):
        self.line, self.col = line, col
        where = f"line {line}" + (f", column {col}" if col else "") + ": " if line else ""
        super().__init__(where + msg)


class CheckError(CierraError):
    """One or more problems found by the checker; `problems` is the list of messages."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("program rejected:\n" + "\n".join(f"  - {p}" for p in self.problems))
