"""Locked-in GJR-GARCH model specification + scale factor."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GarchSpec:
    mean: str = "Constant"
    vol: str = "GARCH"
    p: int = 1
    o: int = 1  # asymmetry (GJR)
    q: int = 1
    dist: str = "t"  # Student's t innovations

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GarchSpec":
        return cls(
            mean=str(d["mean"]),
            vol=str(d["vol"]),
            p=int(d["p"]),
            o=int(d["o"]),
            q=int(d["q"]),
            dist=str(d["dist"]),
        )


DEFAULT_SPEC = GarchSpec()
SCALE_FACTOR = 100.0  # log returns × 100 → percentage for arch's optimizer
