def calculate_mean(data):
    return sum(data) / len(data)


# standard scaler formula z = (x - mean) / std
def calculate_variance(data):
    mean = calculate_mean(data)

    n = len(data)
    result = []

    for x in data:
        ans = (x - mean) ** 2
        result.append(ans)

    return sum(result) / n


def calculate_standard_deviation(data):
    return calculate_variance(data) ** 0.5


def standardize(data, mean, std):
    result = []

    for x in data:
        z = (x - mean) / std
        result.append(z)

    return result


class ManualStandardScaler:
    def fit(self, data):
        self.mean = calculate_mean(data)
        self.std = calculate_standard_deviation(data)

    def transform(self, data):
        return standardize(data, self.mean, self.std)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


if __name__ == "__main__":
    train_distances = [2, 4, 6]
    test_distances = [8, 10]

    # class
    print("\nclass of standard scaler")

    scaler = ManualStandardScaler()

    train_scaled = scaler.fit_transform(train_distances)
    test_scaled = scaler.transform(test_distances)

    print("\nTrain Standard scaler:", train_scaled)
    print("\nTest standard scaler:", test_scaled)
