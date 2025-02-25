# preprocess.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, file_path: str):
        """
        Initialize with the path to the raw CSV data file.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the raw CSV dataset.
        """
        self.df = pd.read_csv(self.file_path)
        print("Initial data preview:")
        print(self.df.head())
        print(self.df.info())
        print(self.df.describe())
        print("Data shape:", self.df.shape)
        return self.df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Clean the data, perform feature engineering, and visualize key distributions.
        """
        if self.df is None:
            self.load_data()
        df = self.df.copy()

        # Check for duplicate rows
        duplicates = df.duplicated()
        if duplicates.any():
            print("Duplicate rows found:")
            print(df[duplicates])
        print(f"Number of duplicate rows: {duplicates.sum()}")

        # Select relevant features for the model
        df = df[['model_year', 'make', 'model', 'mileage', 'price',
                 'transmission_from_vin', 'fuel_type_from_vin', 'days_on_market',
                 'msrp', 'number_price_changes', 'dealer_name', 'listing_type',
                 'listing_first_date']]
        print("Selected columns and their data types:")
        print(df.dtypes)

        # Count the number of price values between 0 and 1000
        price_range_count = df[(df['price'] >= 0) & (df['price'] <= 1000)].shape[0]
        print(f'Number of price values between 0 and 1000: {price_range_count}')

        # Replace low price values with the group mean price
        mean_prices = df[df['price'] > 1000].groupby(
            ['make', 'model', 'model_year', 'transmission_from_vin', 'fuel_type_from_vin']
        )['price'].mean().reset_index()
        mean_prices.rename(columns={'price': 'mean_price'}, inplace=True)
        df = df.merge(mean_prices,
                      on=['make', 'model', 'model_year', 'transmission_from_vin', 'fuel_type_from_vin'],
                      how='left')
        df['price'] = df.apply(
            lambda row: row['mean_price'] if 0 <= row['price'] <= 1000 else row['price'], axis=1)
        df.drop(columns=['mean_price'], inplace=True)
        price_range_count = df[(df['price'] >= 0) & (df['price'] <= 1000)].shape[0]
        print(f'Number of price values between 0 and 1000 after replacement: {price_range_count}')

        # Repeat the same process for 'msrp'
        msrp_range_count = df[(df['msrp'] >= 0) & (df['msrp'] <= 1000)].shape[0]
        print(f'Number of msrp values between 0 and 1000: {msrp_range_count}')
        mean_values = df[(df['price'] > 1000) & (df['msrp'] > 1000)].groupby(
            ['make', 'model', 'model_year', 'transmission_from_vin', 'fuel_type_from_vin']
        )[["price", "msrp"]].mean().reset_index()
        mean_values.rename(columns={'price': 'mean_price', 'msrp': 'mean_msrp'}, inplace=True)
        df = df.merge(mean_values,
                      on=['make', 'model', 'model_year', 'transmission_from_vin', 'fuel_type_from_vin'],
                      how='left')
        df['price'] = df.apply(
            lambda row: row['mean_price'] if 0 <= row['price'] <= 1000 else row['price'], axis=1)
        df['msrp'] = df.apply(
            lambda row: row['mean_msrp'] if 0 <= row['msrp'] <= 1000 else row['msrp'], axis=1)
        df.drop(columns=['mean_price', 'mean_msrp'], inplace=True)
        msrp_range_count = df[(df['msrp'] >= 0) & (df['msrp'] <= 1000)].shape[0]
        print(f'Number of msrp values between 0 and 1000 after replacement: {msrp_range_count}')

        # Fill missing values for numeric columns with the mean
        df['price'] = df['price'].fillna(df['price'].mean())
        df['msrp'] = df['msrp'].fillna(df['msrp'].mean())

        print("Missing values after filling in selected columns:")
        print(df[['model_year', 'mileage', 'price', 'msrp', 'make', 'model',
                  'transmission_from_vin', 'fuel_type_from_vin', 'number_price_changes',
                  'dealer_name', 'listing_type']].isnull().sum())

        # Check for missing values in 'transmission_from_vin' and 'fuel_type_from_vin'
        num_missing_transmission = df['transmission_from_vin'].isnull().sum()
        num_missing_fuel_type = df['fuel_type_from_vin'].isnull().sum()
        print(f"Number of NaN values in 'transmission_from_vin': {num_missing_transmission}")
        print(f"Number of NaN values in 'fuel_type_from_vin': {num_missing_fuel_type}")

        # Encode 'transmission_from_vin' column
        df['transmission_from_vin'] = df['transmission_from_vin'].map({'A': 0, 'M': 1, '6': 1, '7': 0})
        print("Encoded 'transmission_from_vin' unique values:", df['transmission_from_vin'].unique())

        # Optional: Plot the distribution of transmissions
        transmission_counts = df['transmission_from_vin'].value_counts()
        transmission_counts.index = transmission_counts.index.map({0: 'A', 1: 'M'})
        plt.figure(figsize=(8, 6))
        plt.pie(transmission_counts, labels=transmission_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Automatic and Manual Cars')
        plt.axis('equal')
        plt.show()

        # Create a new feature: vehicle_age
        current_year = 2024  # Replace with the current year if needed
        df['vehicle_age'] = current_year - df['model_year']

        # Process the 'listing_first_date' column
        df['listing_first_date'] = df['listing_first_date'].astype(str)
        df['listing_first_date'] = df['listing_first_date'].str.replace('/', '-')
        df['listing_first_date'] = pd.to_datetime(df['listing_first_date'], errors='coerce')
        df['year'] = df['listing_first_date'].dt.year
        df['month'] = df['listing_first_date'].dt.month
        df['day'] = df['listing_first_date'].dt.day

        print("Date columns after processing:")
        print(df[['listing_first_date', 'year', 'month', 'day']].head())

        # Save the cleaned DataFrame in the class instance
        self.df = df
        return self.df

    def save_preprocessed_data(self, output_file: str):
        """
        Save the preprocessed DataFrame to a CSV file.
        """
        if self.df is not None:
            self.df.to_csv(output_file, index=False)
            print(f"Preprocessed data saved to {output_file}")
        else:
            print("No data to save. Run preprocess_data() first.")

    def prepare_data(self):
        """
        Split the processed DataFrame into features (X) and target (y).
        """
        if self.df is None:
            raise ValueError("Data not preprocessed yet. Run preprocess_data() first.")
        X = self.df[['number_price_changes', 'vehicle_age', 'mileage', 'price', 'msrp',
                     'dealer_name', 'listing_type', 'make', 'model', 'year', 'month', 'day']]
        y = self.df['days_on_market']
        return X, y

if __name__ == "__main__":
    # Define the raw data file and the output preprocessed file
    RAW_DATA_FILE = 'Data/raw/CBB_Listings cleaned1.csv'
    OUTPUT_FILE = 'Data/preprocessed/preprocessed_data.csv'
    preprocessor = DataPreprocessor(RAW_DATA_FILE)
    preprocessor.load_data()
    preprocessor.preprocess_data()
    preprocessor.save_preprocessed_data(OUTPUT_FILE)
