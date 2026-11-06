from manual_main import delivery_data, pd

delivery_data["km_per_minute"] = (
    delivery_data["distance_km"] / delivery_data["delivery_minutes"]
)

new_delivery = pd.DataFrame({"distance_km": [8], "weather": ["clear"]})

new_delivery["km_per_minute"] = (
    new_delivery["distance_km"] / new_delivery["delivery_minutes"]
)

X = delivery_data[["distance_km", "weather", "km_per_minute"]]

y = delivery_data["delivery_minutes"]

delivery_data["recovered_minutes"] = (
    delivery_data["distance_km"] / delivery_data["km_per_minute"]
)


if __name__ == "__main__":
    print(new_delivery)

    print("\nDelivery data with km_per_minute feature:\n")
    print(delivery_data[["distance_km", "delivery_minutes", "km_per_minute"]])

    print("\nDelivery data with km_per_minute feature:\n")
    print(X.head())

    print("\nDelivery data with recovered_minutes feature:\n")
    print(delivery_data[["delivery_minutes", "recovered_minutes"]])
