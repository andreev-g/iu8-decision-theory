import enum
import yaml
import pydantic
import typing as t


class FuncDirection(str, enum.Enum):
    MIN = "min"
    MAX = "max"


class SimplexProblem(pydantic.BaseModel):
    c: t.List[float]
    A: t.List[t.List[float]]
    b: t.List[float]
    direction: FuncDirection

    @classmethod
    def from_yaml(cls, filename: str) -> "SimplexProblem":
        with open(filename, "r") as f:
            data = yaml.load(f, yaml.CLoader)
            return SimplexProblem(
                c=data["c"],
                A=data["A"],
                b=data["b"],
                direction=data["dir"]
            )
