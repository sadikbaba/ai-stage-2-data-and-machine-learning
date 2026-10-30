def get_unique_categories(data):

    if data == []:
        return []

    cleaned = []

    for x in data:
        if x not in cleaned:
            cleaned.append(x)

    return cleaned


def encode_one(category, categories):
    result = []
    for x in categories:
        if x == category:
            result.append(1)
        else:
            result.append(0)
    return result


def encode_all(data, categories):

    encoded_data = []
    for x in data:
        if is_known_category(x, categories):
            result = encode_one(x, categories)
            encoded_data.append(result)
        else:
            unseen_result = [0] * len(categories)
            encoded_data.append(unseen_result)

    return encoded_data


def is_known_category(category, categories):
    return category in categories


class ManualOneHotEncoder:
    def fit(self, data):
        self.categories = get_unique_categories(data)

    def transform(self, data):
        return encode_all(data, self.categories)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


if __name__ == "__main__":
    training_weather = ["clear", "rain", "clear"]
    test_weather = ["rain", "snow", "clear"]

    # class
    print("class of One Hot Encode\n")
    encoder = ManualOneHotEncoder()

    encoded_training = encoder.fit_transform(training_weather)
    encoded_test = encoder.transform(test_weather)

    print("Categories:", encoder.categories)
    print("Training:", encoded_training)
    print("Testing:", encoded_test)
