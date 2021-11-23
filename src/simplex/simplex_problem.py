import enum
import yaml
import pydantic
import numpy as np
import typing as t


class ComparisonSign(str, enum.Enum):
    EQ = "eq"
    GE = "ge"
    LE = "le"
    GT = "gt"
    LT = "lt"


HUMAN_COMP_SIGNS: t.Dict[ComparisonSign, str] = {
    ComparisonSign.EQ: "==",
    ComparisonSign.GE: ">=",
    ComparisonSign.LE: "<=",
    ComparisonSign.GT: ">",
    ComparisonSign.LT: "<",
}


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
            p = cls(**data)
            assert len(p.c) == len(p.A[0]) and len(p.A) == len(p.b) == len(p.comp_signs), \
                "all list should be same length"
            return p

    def get_canonical(self) -> np.ndarray:
        for i, row in enumerate(self.A):
            if self.comp_signs[i] == ComparisonSign.LE:
                self.A[i] = [
                    -1 * item
                    for item in row
                ]
            elif self.comp_signs[i] == ComparisonSign.GE:
                pass
            else:
                raise ValueError("Logic for this sign is not implemented")
        for i in range(len(self.b)):
            if self.comp_signs[i] == ComparisonSign.GE:
                self.b[i] *= -1
            elif self.comp_signs[i] == ComparisonSign.LE:
                pass
            else:
                raise ValueError("Logic for this sign is not implemented")
        if self.target == FuncTarget.MAX:
            self.c = [
                -1 * c
                for c in self.c
            ]
        elif self.target == FuncTarget.MIN:
            pass
        else:
            raise ValueError("Logic for this target is not implemented")
        for i in range(len(self.A)):
            for j in range(len(self.A[i])):
                self.A[i][j] *= -1
        if self.target == FuncTarget.MIN:
            for i in range(len(self.c)):
                self.c[i] *= -1
        matrix = np.c_[self.b, self.A]
        matrix = np.r_[matrix, [[0, *self.c]]]
        return matrix
