import pandas as pd

delivery_data = pd.read_csv('delivery.csv')

x = delivery_data[['distance_km', 'weather']]
y = delivery_data['delivery_minutes']

if __name__ == '__main__':
    print("\nData:")
    print(delivery_data.head())

    print("\nShape of the data:")
    print(delivery_data.shape)

    print("\nFeatures X:")
    print(x)

    print("\nTarget y:")
    print(y)


  
