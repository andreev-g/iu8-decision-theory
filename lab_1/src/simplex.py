import typing as t
import pandas as pd

from lab_1.src.problem import Problem


class SimplexTable:

    def __init__(self, problem: Problem):
        self._problem = problem
        self._table = self._new_table(problem)

    @staticmethod
    def _new_table(p: Problem) -> pd.DataFrame:
        table: t.List[t.List[float]] = []
        # заполнение ограничений
        for i in range(0, len(p.A)):
            table.append([])
            # свободные члены
            table[i].append(p.b[i])
            # коэффициенты при переменных
            for j in range(0, len(p.A[i]) - len(p.b)):
                table[i].append(p.A[i][j])
        # заполнение коэффициентов ЦФ
        table.append([0])
        for j in range(0, len(p.c) - len(p.b)):
            table[-1].append(-p.c[j])
        # pandas таблица
        return pd.DataFrame(
            data=table,
            index=pd.Index(data=["x4", "x5", "x6", "F"]),
            columns=["Si0", "x1", "x2", "x3"]
        )

    def _swap_basic_vars(self, r: int, k: int) -> None:
        new_columns = self._table.columns.array
        new_indices = self._table.index.array

        new_columns[r], new_indices[k] = new_indices[k], new_columns[r]

        # создание новой таблицы
        new_table = [[0 for _ in range(0, len(self._table.iloc[i]))] for i in range(0, len(self._table))]

        # применение правил создания новой таблицы
        new_table[r][k] = 1 / self._table.iloc[r][k]
        for j in range(0, len(self._table.iloc[r])):
            if j != k:
                new_table[r][j] = self._table.iloc[r][j] / self._table.iloc[r][k]
        for i in range(0, len(self._table)):
            if i != r:
                new_table[i][k] = - self._table.iloc[i][k] / self._table.iloc[r][k]
        for i in range(0, len(self._table)):
            for j in range(0, len(self._table.iloc[i])):
                if i != r and j != k:
                    new_table[i][j] = self._table.iloc[i][j] - self._table.iloc[i][k] * self._table.iloc[r][j] / self._table.iloc[r][k]

        self._table = pd.DataFrame(
            data=new_table,
            index=new_indices,
            columns=new_columns
        )

    def find_bearing_resolve(self) -> None:
        def find_resolving_elem(df: pd.DataFrame):
            """
            Функция поиска разрешающего элемента
            """
            resolving_column = None
            # нахождение разрешающей строки путем анализа свободных членов и
            # коэффициентов при свободных переменных
            for i in range(0, len(df) - 1):
                if df[i][0] < 0:
                    # если свободный член i отрицательный, то
                    # ищем отрицательный элемент ij
                    for j in range(0, len(df[i])):
                        if df[i][j] < 0:
                            resolving_column = j
                    # если отрицательный элемент ij не найден,
                    # то заключаем, что система несовместна
                    if resolving_column is None:
                        raise ValueError('Система несовместна!')
                    break

            # ищем разрешающую строку, находя минимальной частное si0/sik
            min_division = None
            resolving_row = None
            for i in range(0, len(df) - 1):
                if df[i][resolving_column] != 0:
                    division = df[i][0] / df[i][resolving_column]
                    if division > 0 and (min_division is None or division < min_division):
                        min_division = division
                        resolving_row = i

            return resolving_row, resolving_column

        # производим проверку на то, является ли решение опорным,
        # и возвращаемся из функции
        # если оно не опроное, то заменяем одну базисную переменную
        # с печатью промежуточного результата
        step = 0
        while True:
            step += 1
            if self.is_bearing_resolve():
                return

            ij = find_resolving_elem(self._table)
            self._swap_basic_vars(*ij)

    def find_optimal_resolve(self):
        def find_resolving_elem(df: pd.DataFrame):
            """
            Функция поиска разрешающего элемента
            """
            # нахождение разрешающей строки путем анализа коэффициентов при ЦФ
            # разрешающий столбец располагается там, где коэффициент при свободной
            # переменной положителен (что говорит о неоптимальности данного решения)
            resolving_column = None
            for j in range(1, len(df.iloc[-1])):
                value = df.iloc[-1, j]
                if value > 0:
                    resolving_column = j
                    break

            # нахождение минимального положительного частного вида si0/sik
            min_division = None
            resolving_row = None
            for i in range(0, len(df) - 1):
                if df.iloc[i][resolving_column] != 0:
                    division = df.iloc[i][0] / df.iloc[i][resolving_column]
                    if division > 0 and (min_division is None or division < min_division):
                        min_division = division
                        resolving_row = i

            return resolving_row, resolving_column

        # производим проверку на то, является ли решение оптимальным,
        # и возвращаемся из функции
        # если оно не оптимально, то заменяем одну базисную переменную
        # с печатью промежуточного результата
        step = 0
        while True:
            step += 1
            if self.is_optimal_resolve():
                return

            ij = find_resolving_elem(self._table)
            self._swap_basic_vars(*ij)

    def is_bearing_resolve(self) -> bool:
        """
        Поиск опорного решения
        """
        for s in self._get_si0_list():
            if s < 0:
                return False
        return True

    def is_optimal_resolve(self) -> bool:
        for v in self._get_f_list()[1:]:
            if v > 0:
                return False
        return True

    def _get_f_list(self) -> t.List[float]:
        return self._table.iloc[-1, :]

    def _get_si0_list(self) -> t.List[float]:
        return self._table.iloc[:, 0]
