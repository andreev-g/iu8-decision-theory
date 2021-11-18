import typing as t
import pandas as pd
from tabulate import tabulate


class SimplexTable(pd.DataFrame):

    _F = "F"
    _Si0 = "si0"
    _ROW = "row"
    _COL = "column"

    NO_SOLUTIONS_ERR_MSG = "there aren't solutions"

    _target: str = None

    def __init__(
            self,
            canonical_start_table: t.List[t.List[float]],
            target: str
    ):
        if target not in ("min", "max"):
            raise ValueError("system's target should be one of: (min, max)")
        self._target = target
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

    def find_base_solution(self, inplace: bool = False) -> 'SimplexTable':
        simplex: SimplexTable = self._get_self(make_copy=not inplace)
        while not simplex._is_base_solution():
            row, col = simplex._get_pivot_indices(start_row=None)
            simplex._swap_vars(row, col)
        return simplex

    def find_optimal_solution(self, inplace: bool = False) -> 'SimplexTable':
        simplex = self._get_self(make_copy=not inplace)
        while not simplex._is_optimal_solution():
            row, col = simplex._get_pivot_indices(start_row=self._F)
            simplex._swap_vars(row, col)
            simplex.print()
            import time
            time.sleep(1)
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

    def _is_base_solution(self) -> bool:
        for row in self.index.copy().drop("F"):
            if self.loc[row, self._Si0] < 0:
                assert any(self.loc[row].iloc[1:] < 0), self.NO_SOLUTIONS_ERR_MSG
                return False
        return True

    def _is_optimal_solution(self) -> bool:
        if self._target == "min":
            return all(self.loc[self._F].drop(self._Si0) < 0)
        return all(self.loc[self._F].drop(self._Si0) > 0)

    def _get_pivot_indices(self, start_row: t.Optional[str]) -> t.Tuple[str, str]:
        if start_row is None:
            start_row = self.loc[:, self._Si0].idxmin()
        start_row_xi = self.loc[start_row].drop(self._Si0)
        assert any(start_row_xi < 0), self.NO_SOLUTIONS_ERR_MSG
        col = start_row_xi.idxmin()
        row = (self.loc[:, self._Si0].drop("F") / self.loc[:, col].drop("F")).idxmin()
        return row, col

    def _get_self(self, make_copy: bool) -> 'SimplexTable':
        if make_copy:
            return self.copy()
        return self
