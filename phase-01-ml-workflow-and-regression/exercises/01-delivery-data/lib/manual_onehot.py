def get_unique_categories(data):

    if data == []:
        return []
       
    cleaned =[]

    for x in data:
       if x not in cleaned:
           cleaned.append(x)
           
    return cleaned

weather = ["clear", "rain", "clear", "rain"]

print(get_unique_categories(weather))
print(get_unique_categories(["clear", "rain", "snow"]))



def encode_one(category, categories): 
    result = []
    for x in categories:
        if x == category:
            result.append(1)
        else:
            result.append(0)
    return result

    

categories = ["clear", "rain", "snow"]
print(encode_one("rain", categories))

def encode_all(data):
    
    categories = get_unique_categories(data)
    encoded_data = []
    for x in data:
       result = encode_one(x, categories)
       encoded_data.append(result)
    return encoded_data

weather = ["clear", "rain", "clear", "rain"]
print(encode_all(weather)) # result = [[1, 0], [0, 1], [1, 0], [0, 1]]

