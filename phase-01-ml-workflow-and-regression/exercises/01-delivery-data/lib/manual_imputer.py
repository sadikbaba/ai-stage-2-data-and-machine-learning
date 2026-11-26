import math


class ManualSimpleImputer:
    def __init__(self, strategy=None):
        self.strategy = strategy

    def is_missing(self, value):
        return value is None or (isinstance(value, float) and math.isnan(value))

    def get_column(self, data, j):
        column = []
        for i in range(len(data)):
            if not self.is_missing(data[i][j]):
                column.append(data[i][j])
        return column

    def mean(self, data):
        if not data:
            return []
        self.result = []
        for j in range(len(data[0])):
            column = self.get_column(data, j)
            total = 0
            for value in column:
                total += value
            self.result.append(total / len(column))
        return self

    def median(self, data):
        if not data:
            return []
        self.result = []
        for j in range(len(data[0])):
            column = sorted(self.get_column(data, j))
            n = len(column)
            middle = n // 2
            if n % 2 == 1:
                self.result.append(column[middle])
            else:
                self.result.append((column[middle - 1] + column[middle]) / 2)
        return self

    def most_frequent(self, data):
        if not data:
            return []
        self.result = []
        for j in range(len(data[0])):
            column = self.get_column(data, j)
            counts = {}
            for value in column:
                if value in counts:
                    counts[value] += 1
                else:
                    counts[value] = 1

            best_value = None
            best_count = 0
            for value, count in counts.items():
                if count > best_count or (count == best_count and value < best_value):
                    best_value = value
                    best_count = count
            self.result.append(best_value)
        return self

    def fit(self, data, y=None):
        if self.strategy == "mean":
            return self.mean(data)
        if self.strategy == "median":
            return self.median(data)
        if self.strategy == "most_frequent":
            return self.most_frequent(data)
        if self.strategy == "constant":
            self.result = [0] * len(data[0])
            return self
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def transform(self, data):
        new_data = []
        for row in data:
            new_row = []
            for j in range(len(row)):
                if self.is_missing(row[j]):
                    new_row.append(self.result[j])
                else:
                    new_row.append(row[j])
            new_data.append(new_row)
        return new_data

    def fit_transform(self, data, y=None):
        self.fit(data)
        return self.transform(data)


if __name__ == "__main__":
    data = [[2], [4], [6], [100], [8], [2]]
    for s in ["mean", "median", "most_frequent", "constant"]:
        imputer = ManualSimpleImputer(strategy=s)
        imputer.fit(data)
        print(s, imputer.result)

    missing = [[2], [None], [8], [10], [3], [7]]
    imputer = ManualSimpleImputer(strategy="median")
    print(imputer.fit_transform(missing))


    new_data = [
    [2, "clear"],
    [None, "rain"],
    [8, "clear"],
    [10, None],
    [3, "rain"],
]
    print("\n")
    imputer = ManualSimpleImputer(strategy="most_frequent")
    imputer.fit(new_data)
    print("most frequent:", imputer.transform(new_data))

    new_data = [
        [2],
        [None],
        [8],
        [10],
        [3],
    ]
    print("\n")
    imputer = ManualSimpleImputer(strategy="median")
    imputer.fit(new_data)
    print("median:", imputer.transform(new_data))