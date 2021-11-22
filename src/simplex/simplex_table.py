import typing as t
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.simplex.simplex_problem import FuncTarget


class SimplexTable(pd.DataFrame):

    _F = "F"
    _Si0 = "si0"
    _ROW = "row"
    _COL = "column"

    NO_SOLUTIONS_ERR_MSG = "there aren't solutions"

    _target: str = None
    _c: t.List[float] = None

    def __init__(
            self,
            canonical_start_table: np.ndarray,
            target: FuncTarget,
            c: t.List[float]
    ):
        if target not in ("min", "max"):
            raise ValueError("system's target should be one of: (min, max)")
        self._target = target
        self._c = c
        minor_vars_num = len(canonical_start_table[0]) - 1
        basis_vars_num = len(canonical_start_table) - 1
        columns = [self._Si0] + [
            f"x{i}"
            for i in range(1, minor_vars_num + 1)
        ]
        index = [
            f"x{i + minor_vars_num}"
            for i in range(1, basis_vars_num + 1)
        ] + [self._F]
        super().__init__(
            data=canonical_start_table,
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
        while True:
            if simplex.is_base_solution():
                break
            row, col = simplex._get_base_pivot_indices()
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
        while True:
            if simplex._is_optimal_solution():
                break
            row, col = simplex._get_opti_pivot_indices()
            simplex._swap_vars(row, col)
            if print_logs:
                print(f"Разрешающие (строка, столбец) : ({row} , {col})")
                simplex.print()
                print()
                print("~" * 70 + "\n")
        return simplex

    def print(self) -> None:
        print(tabulate(self, headers="keys", tablefmt="psql"))

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
        for row in self.index.copy().drop(self._F):
            if self.loc[row, self._Si0] < 0:
                assert any(self.loc[row].iloc[1:] < 0), self.NO_SOLUTIONS_ERR_MSG
                return False
        return True

    def _is_optimal_solution(self) -> bool:
        if self._target == FuncTarget.MIN:
            return all(self.loc[self._F].drop(self._Si0) < 0)
        return all(self.loc[self._F].drop(self._Si0) > 0)

    def _get_base_pivot_indices(self) -> t.Tuple[str, str]:
        start_row = self.loc[:, self._Si0].drop(self._F).idxmin()
        start_row_xi = self.loc[start_row].drop(self._Si0)
        assert any(start_row_xi < 0), self.NO_SOLUTIONS_ERR_MSG
        col = start_row_xi.idxmin()
        row = (self.loc[:, self._Si0].drop(self._F) / self.loc[:, col].drop(self._F)).idxmin()
        return row, col

    def _get_opti_pivot_indices(self) -> t.Tuple[str, str]:
        col = None
        for col_name, f in self.loc[self._F, :].drop(self._Si0).items():
            if (f > 0 and self._target == FuncTarget.MIN) or (f < 0 and self._target == FuncTarget.MAX):
                col = col_name
                break
        assert col is not None, "There are not positives in F"
        si0_col_ratios = self.loc[:, self._Si0].drop(self._F) / self.loc[:, col].drop(self._F)
        si0_col_ratios.drop(
            labels=[label for label, value in si0_col_ratios.items() if value < 0],
            inplace=True
        )
        row = si0_col_ratios.idxmin()
        return row, col

    def check_solution(self) -> bool:
        solution = self.get_solution()
        simplex_f = - round(self.loc[self._F, self._Si0], 3)
        calculated_f = round(sum(solution[i] * self._c[i] for i in range(len(self._c))), 3)
        print(" + ".join(
            f"{round(solution[i], 3)} * {round(self._c[i], 3)}" for i in range(len(self._c))
        ) + f" == {simplex_f}")
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
