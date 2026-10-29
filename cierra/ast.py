"""Syntax tree of a Cierra program. Values are plain Python: float/int/bool/str (identifiers are Ident), lists,
Range and Keys."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Ident:
    name: str


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float


@dataclass(frozen=True)
class Keys:
    interp: str
    keys: tuple  # ((frame, value), ...)


@dataclass
class Call:
    name: str
    args: dict            # name -> value (insertion order kept)
    line: int = 0


# masks
@dataclass
class MaskCall:
    call: Call


@dataclass
class MaskNot:
    m: object


@dataclass
class MaskBin:
    op: str  # "and" | "or" | "+" | "-"
    a: object
    b: object


@dataclass
class MaskRef:
    """key(name, gain, offset). Of a node: that node's key as it was made on that node's own input, carried here
    (carried True). Of a `key` statement: that mask, made here on this node's input. The checker fills target
    with the referenced (checked) mask expression and sets carried."""
    name: str
    gain: object = 1.0
    offset: object = 0.0
    line: int = 0
    target: object = None
    carried: bool = False


@dataclass
class Key:
    """`key name = mask`: a named mask that grades nothing itself; nodes use it with key(name)."""
    id: str
    mask: object
    line: int = 0


# graph expressions
@dataclass
class Ref:
    name: str
    line: int = 0


@dataclass
class Pass:
    """`pass`: the graph term's input, unchanged (a layer bottom or a parallel branch that keeps the picture)."""


@dataclass
class Serial:
    items: list


@dataclass
class Parallel:
    branches: list


@dataclass
class Layer:
    bottom: object
    top: object
    mode: str = "normal"
    opacity: object = 1.0


@dataclass
class Node:
    id: str
    call: Call
    mask: object = None
    mix: object = 1.0
    label: str = ""
    enabled: bool = True
    line: int = 0


@dataclass
class Program:
    version: str
    name: str = ""
    about: str = ""
    input: str = "linear_rec709"
    output: Call = None
    reference_width: int = 1920
    fps: float = 24.0
    seed: int = 0
    nodes: dict = field(default_factory=dict)   # id -> Node, in declaration order
    graph: object = None
    photo_graph: object = None
    requires: tuple = ()
    keys: dict = field(default_factory=dict)    # id -> Key, in declaration order
