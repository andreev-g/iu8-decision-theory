import enum

import numpy as np
import yaml
import pydantic
import typing as t


class ComparisonSign(str, enum.Enum):
    EQ = "eq"
    GE = "ge"
    LE = "le"
    GT = "gt"
    LT = "lt"


class FuncTarget(str, enum.Enum):
    MIN = "min"
    MAX = "max"


class SimplexProblem(pydantic.BaseModel):
    c: t.List[float]
    A: t.List[t.List[float]]
    b: t.List[float]
    comp_signs: t.List[ComparisonSign]
    target: FuncTarget

    @classmethod
    def from_yaml(cls, filename: str) -> "SimplexProblem":
        with open(filename, "r") as f:
            data = yaml.load(f, yaml.CLoader)
            p = SimplexProblem(
                c=data["c"],
                A=data["A"],
                b=data["b"],
                comp_signs=data["comparison_signs"],
                target=data["dir"]
            )
            assert len(p.c) == len(p.A) == len(p.b) == len(p.comp_signs), "all list should be same length"
            return p

    def to_canonical(self) -> None:
        pass

    def get_matrix(self) -> np.ndarray:
        matrix = np.c_[self.b, self.A]
        matrix = np.r_[matrix, [[0, *self.c]]]
        return matrix
