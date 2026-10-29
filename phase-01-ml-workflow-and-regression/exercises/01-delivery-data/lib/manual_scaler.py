def calculate_mean(data):
    return sum(data)/len(data)

# standard scaler formula z = (x - u) / s
def calculate_variance(data):
    mean = calculate_mean(data)

    n = len(data)
    result = []

    for x in data:
       ans = (x - mean) **2
       result.append(ans)

    return sum(result)/n

def calculate_standard_deviation(data):
    return calculate_variance(data) ** 0.5


def standardize(data, mean, std):
    result = []

    for x in data:
        z = (x - mean) / std
        result.append(z)

    return result

if __name__ == "__main__":
    train_distances = [2, 4, 6]
    test_distances = [8 , 10]

    training_data_mean = calculate_mean(train_distances)
    training_data_std = calculate_standard_deviation(train_distances)

    training_standard_scaler = standardize(train_distances, training_data_mean, training_data_std)

    test_standard_scaler = standardize(test_distances, training_data_mean, training_data_std)


    print("\nTraining Standard scaler:", training_standard_scaler)
    print("\nTest standard scaler:" , test_standard_scaler)