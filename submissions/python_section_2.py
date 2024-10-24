import pandas as pd

def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the distances in the provided CSV file.

    Args:
        file_path (str): Path to the CSV file containing distances.

    Returns:
        pandas.DataFrame: Distance matrix with rows and columns sorted in ascending order.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces

    # Ensure the expected column names are present
    if 'id_start' not in df.columns or 'id_end' not in df.columns or 'distance' not in df.columns:
        raise ValueError("CSV must contain 'id_start', 'id_end', and 'distance' columns")

    # Create a set of unique locations and convert to a sorted list
    locations = sorted(set(df['id_start']).union(set(df['id_end'])))
    
    # Initialize the distance matrix with infinity
    distance_matrix = pd.DataFrame(index=locations, columns=locations).fillna(float('inf'))
    
    # Set the diagonal to 0
    for location in locations:
        distance_matrix.at[location, location] = 0

    # Populate the distance matrix with known distances
    for _, row in df.iterrows():
        from_loc = row['id_start']
        to_loc = row['id_end']
        distance = row['distance']
        
        # Update the distance matrix for both directions (ensuring symmetry)
        distance_matrix.at[from_loc, to_loc] = min(distance_matrix.at[from_loc, to_loc], distance)
        distance_matrix.at[to_loc, from_loc] = min(distance_matrix.at[to_loc, from_loc], distance)

    # Apply the Floyd-Warshall algorithm to calculate cumulative distances
    for k in locations:
        for i in locations:
            for j in locations:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix



def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): Distance matrix DataFrame.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Create an empty list to hold the rows of the new DataFrame
    unrolled_data = []

    # Iterate over each row and column in the distance matrix
    for id_start in df.index:
        for id_end in df.columns:
            # Skip the case where id_start is the same as id_end
            if id_start != id_end:
                distance = df.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a new DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df



def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame containing columns 'id_start', 'id_end', and 'distance'.
        reference_id (int): The ID for which to find similar average distances.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate the average distance for the reference ID
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    # Calculate the thresholds (10% above and below the average)
    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1

    # Calculate average distances for all IDs in the DataFrame
    average_distances = df.groupby('id_start')['distance'].mean().reset_index()
    average_distances.columns = ['id_start', 'average_distance']

    # Filter IDs within the 10% threshold of the reference ID's average distance
    filtered_ids = average_distances[
        (average_distances['average_distance'] >= lower_threshold) &
        (average_distances['average_distance'] <= upper_threshold)
    ]

    # Sort the resulting DataFrame by average distance
    sorted_result = filtered_ids.sort_values(by='average_distance')

    return sorted_result



def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates based on vehicle types and add them as new columns to the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing columns 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: Updated DataFrame with added columns for toll rates for different vehicle types.
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type and add as new columns
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient
    
    # Drop the 'distance' column
    # df = df.drop(columns=['distance'])

    return df



import numpy as np
from datetime import time, timedelta

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day and add them to the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing toll rates for different vehicle types and distances.

    Returns:
        pandas.DataFrame: Updated DataFrame with added columns for time-based toll rates.
    """
    # Define the time intervals and discount factors
    time_intervals = {
        'weekday': [
            (time(0, 0), time(10, 0), 0.8),
            (time(10, 0), time(18, 0), 1.2),
            (time(18, 0), time(23, 59, 59), 0.8)
        ],
        'weekend': [
            (time(0, 0), time(23, 59, 59), 0.7)
        ]
    }
    
    # Create a list to store the new rows
    new_data = []

    # Loop through each unique (id_start, id_end) pair
    for (id_start, id_end), group in df.groupby(['id_start', 'id_end']):
        distance = group['distance'].values[0]  # Assuming distance is the same for each group
        
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            for interval in time_intervals['weekday' if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else 'weekend']:
                start_time, end_time, discount_factor = interval
                
                # Create a new row with calculated toll rates
                row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                }
                
                # Calculate toll rates for each vehicle type and round to 2 decimal places
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    row[vehicle] = round(group[vehicle].mean() * discount_factor, 2)  # Round to 2 decimal places
                
                new_data.append(row)

    # Convert the list to a DataFrame
    time_based_toll_df = pd.DataFrame(new_data)

    return time_based_toll_df

# # Example usage (assuming you have the distance matrix from the previous function)
# distance_matrix = calculate_distance_matrix(file_path)
# print(distance_matrix)
# # Print unrolled_df
# unrolled_df = unroll_distance_matrix(distance_matrix)
# print(unrolled_df)
# # Example usage (assuming you have the unrolled DataFrame from Question 10)
# result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id=1001400)
# print(result_df)
# print()
# # print toll rate
# toll_rate_df = calculate_toll_rate(unrolled_df)
# print(toll_rate_df)
# print()
# # print time based toll rate
# time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)
# print(time_based_toll_df)