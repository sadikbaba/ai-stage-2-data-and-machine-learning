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


if __name__ == "__main__":

    training_weather = ["clear", "rain", "clear"]
    test_weather = ["rain", "snow","clear" ]

    training_categories = get_unique_categories(training_weather)
  
    encoded_training = encode_all(training_weather, training_categories)
    encoded_weather = encode_all(test_weather, training_categories)

    print("Training:",encoded_training)
    print("\nTesting:",encoded_weather) 