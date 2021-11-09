import yaml
import pydantic
import typing as t


class Problem(pydantic.BaseModel):
    c: t.List[float]
    A: t.List[t.List[float]]
    b: t.List[float]

    @classmethod
    def from_yaml(cls, filename: str) -> "Problem":
        with open(filename, "r") as f:
            data = yaml.load(f, yaml.CLoader)
            return Problem(
                c=data["c"],
                A=data["A"],
                b=data["b"]
            )

    def to_canonical(self) -> "Problem":
        """
        from:
            F = cx -> max
            Ax <= b
            x >= 0
        to:
            F = -cx -> min
            A(x + [x_i]) = b
            x >= 0
        """
        # вычисляем количество фиктивных переменных
        fictious_vars = len(self.A)
        # переводим коэффициенты при линейной комбинации ЦФ в канонический вид
        # также добавляем нулевые коэффициенты при фиктивных переменных
        canonical_c = []
        for ci in self.c:
            canonical_c.append(-ci)
        for _ in range(0, fictious_vars):
            canonical_c.append(0)
        # переводим матрицу ограничений в канонический вид
        canonical_A = []
        for i in range(0, len(self.A)):
            # формирование строк
            canonical_A.append([])

            # копирование коффициентов для реальных переменных
            for j in range(0, len(self.A[i])):
                canonical_A[i].append(self.A[i][j])
            # создание коффициентов для фиктивных переменных
            for k in range(0, fictious_vars):
                if k == i:
                    canonical_A[i].append(1)
                else:
                    canonical_A[i].append(0)
        return Problem(
            c=canonical_c,
            A=canonical_A,
            b=self.b
        )
