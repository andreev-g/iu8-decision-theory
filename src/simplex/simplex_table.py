import typing as t
import pandas as pd
from tabulate import tabulate

from src.simplex.simplex_problem import (
    SimplexProblem,
    HUMAN_COMP_SIGNS
)


class SimplexTable(pd.DataFrame):

    _F = "F"
    _Si0 = "si0"
    _ROW = "row"
    _COL = "column"

    NO_SOLUTIONS_ERR_MSG = "there aren't solutions"

    _problem: SimplexProblem = None

    def __init__(
            self,
            problem: SimplexProblem
    ):
        self._problem = problem.copy()
        canonical_matrix = problem.get_canonical()
        minor_vars_num = len(canonical_matrix[0]) - 1
        basis_vars_num = len(canonical_matrix) - 1
        columns = [self._Si0] + [
            f"x{i}"
            for i in range(1, minor_vars_num + 1)
        ]
        index = [
            f"x{i + minor_vars_num}"
            for i in range(1, basis_vars_num + 1)
        ] + [self._F]
        super().__init__(
            data=canonical_matrix,
            index=index,
            columns=columns,
            dtype=float,
            copy=True
        )

    def find_base_solution(
            self,
            inplace: bool = False,
            print_logs: bool = False
    ) -> 'SimplexTable':

        simplex: SimplexTable = self._get_self(make_copy=not inplace)

        while not simplex.is_base_solution():

            row, col = simplex._get_base_indices()
            simplex._swap_vars(row, col)

            if print_logs:
                print()
                print("~" * 70 + "\n")
                print(f"Разрешающие (строка, столбец) : ({row} , {col})")
                simplex.print()
        return simplex

    def find_optimal_solution(
            self,
            inplace: bool = False,
            print_logs: bool = False
    ) -> 'SimplexTable':

        simplex = self._get_self(make_copy=not inplace)

        while not simplex.is_optimal_solution():

            row, col = simplex._get_optimal_indices()
            simplex._swap_vars(row, col)

            if print_logs:
                print(f"Разрешающие (строка, столбец) : ({row} , {col})")
                simplex.print()
                print()
                print("~" * 70 + "\n")
        return simplex

    def print(self) -> None:
        print(
            tabulate(
                self.applymap(lambda x: x if x != 0 else 0.),
                headers="keys",
                tablefmt="psql"
            )
        )

    def _swap_vars(self, row: str, col: str) -> None:
        self._check_swap_index(row, loc=self._ROW)
        self._check_swap_index(col, loc=self._COL)

        s_rk = 1 / self.loc[row, col]
        s_rj = self.loc[row] / self.loc[row, col]
        s_ik = -1 * self.loc[:, col] / self.loc[row, col]
        self.loc[:, :] = [
            [
                self.iloc[i, j] - self.loc[:, col].iloc[i] * self.loc[row].iloc[j] / self.loc[row, col]
                for j in range(len(self.columns))
            ]
            for i in range(len(self.index))
        ]
        self.loc[row, :] = s_rj
        self.loc[:, col] = s_ik
        self.loc[row, col] = s_rk

        self.rename(columns={col: row}, inplace=True)
        self.rename(index={row: col}, inplace=True)

    def _check_swap_index(self, name: str, loc: str) -> None:
        if loc not in (self._ROW, self._COL):
            raise ValueError(f"please, specify one of (\"{self._ROW}\", \"{self._COL}\"); passed: {loc}")
        sequence = self.columns if loc == self._COL else self.index
        if name not in sequence:
            raise IndexError(f"No such value in {loc}: {name}")
        if name in (self._F, self._Si0):
            raise ValueError(f"Not allowed to access {loc} value: {name}")

    def is_base_solution(self) -> bool:
        return all(self.loc[:, self._Si0].drop(self._F) >= 0)

    def is_optimal_solution(self) -> bool:
        return all(self.loc[self._F].drop(self._Si0) <= 0)

    def _get_base_indices(self):
        col = None
        for basis_x in self.index.drop(self._F):
            if self.loc[basis_x, self._Si0] >= 0:
                continue
            for free_x in self.columns.drop(self._Si0):
                if self.loc[basis_x, free_x] < 0:
                    col = free_x

            if not col:
                raise ValueError("No solutions!")

        row = None
        min_div = None
        for basis_x in self.index.drop(self._F):
            if self.loc[basis_x, col] == 0:
                continue
            div = self.loc[basis_x, self._Si0] / self.loc[basis_x, col]
            if div > 0 and (min_div is None or div < min_div):
                min_div = div
                row = basis_x

        if not (row and col):
            raise ValueError("No solutions!")

        return row, col

    def _get_optimal_indices(self) -> t.Tuple[str, str]:
        col = None
        for free_x in self.columns.drop(self._Si0):
            if self.loc[self._F, free_x] > 0:
                col = free_x

        row = None
        min_div = None
        for basis_x in self.index.drop(self._F):
            if self.loc[basis_x, col] == 0:
                continue
            div = self.loc[basis_x, self._Si0] / self.loc[basis_x, col]
            if div > 0 and (min_div is None or div < min_div):
                min_div = div
                row = basis_x

        return row, col

    def check_solution(self) -> bool:
        solution = self.get_solution()
        simplex_f = round(self.loc[self._F, self._Si0], 3)
        calculated_f = round(sum(solution[i] * self._problem.c[i] for i in range(len(self._problem.c))), 3)
        print("F: " + " + ".join(
            f"{round(solution[i], 3)} * {round(self._problem.c[i], 3)}" for i in range(len(self._problem.c))
        ) + f" == {simplex_f}")
        for i, row in enumerate(self._problem.A):
            comp_sign = HUMAN_COMP_SIGNS[self._problem.comp_signs[i]]
            print(f"Условие {i + 1}: " + " + ".join(
                f"{round(solution[j], 3)} * {round(a)}" for j, a in enumerate(row)
            ) + f" == {round(sum(solution[j] * a for j, a in enumerate(row)), 3)} {comp_sign} {self._problem.b[i]}")
        return simplex_f == calculated_f

    def get_solution(self) -> t.List[float]:
        return [
            0 if f"x{i}" not in self.index else self.loc[f"x{i}", self._Si0]
            for i in range(1, len(self))
        ]

    def _get_self(self, make_copy: bool) -> 'SimplexTable':
        if make_copy:
            return self.copy()
        return self
