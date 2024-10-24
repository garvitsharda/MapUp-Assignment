from typing import Dict, List
import pandas as pd
import numpy as np



def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    # Iterate over the list in steps of n
    for i in range(0, len(lst), n):
        # Get the current group of n elements
        group = lst[i:i+n]
        # Reverse the group manually without slicing or using reverse()
        reversed_group = []
        for j in range(len(group) - 1, -1, -1):
            reversed_group.append(group[j])
        # Replace the original group with the reversed one
        lst[i:i+n] = reversed_group
    return lst



def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)

    sorted_dict = dict(sorted(length_dict.items()))
    return sorted_dict



def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flat_dict = {}

    def flatten(current_dict, parent_key=''):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                flatten(value, new_key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flatten(item, f"{new_key}[{i}]")
                    else:
                        flat_dict[f"{new_key}[{i}]"] = item
            else:
                flat_dict[new_key] = value

    flatten(nested_dict)
    return flat_dict



def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start=0):
        if start == len(nums):
            result.append(nums[:])
            return
        
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] not in seen:
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]  # Swap
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]  # Swap back

    result = []
    nums.sort()  # Sort to handle duplicates
    backtrack()
    return result
    pass



import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    
    # Define regex patterns for the three date formats
    patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',  # dd-mm-yyyy
        r'\b(\d{2})/(\d{2})/(\d{4})\b',  # mm/dd/yyyy
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'   # yyyy.mm.dd
    ]
    
    found_dates = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Reconstruct the date in its original format
            if len(match) == 3:  # Ensure match has three groups
                if pattern == patterns[0]:  # dd-mm-yyyy
                    found_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
                elif pattern == patterns[1]:  # mm/dd/yyyy
                    found_dates.append(f"{match[0]}/{match[1]}/{match[2]}")
                elif pattern == patterns[2]:  # yyyy.mm.dd
                    found_dates.append(f"{match[0]}.{match[1]}.{match[2]}")
    
    return found_dates



import polyline
import math
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude,
    and distance between consecutive points using the Haversine formula.
    
    Args:
        polyline_str (str): The encoded polyline string.
    
    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    
    Example:
        >>> encoded = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
        >>> df = polyline_to_dataframe(encoded)
        >>> print(df)
           latitude  longitude    distance
        0  38.5000  -120.2000     0.0000
        1  40.7000  -120.9500  249982.85
        2  43.2520  -126.4530  628236.56
    """
    try:
        # Decode polyline into list of coordinates
        coordinates = polyline.decode(polyline_str)
        
        # Initialize lists for DataFrame columns
        latitudes = []
        longitudes = []
        distances = []
        
        # First point has no previous point, so distance is 0
        if coordinates:
            latitudes.append(coordinates[0][0])
            longitudes.append(coordinates[0][1])
            distances.append(0)
        
        # Process remaining points
        for i in range(1, len(coordinates)):
            # Add coordinates to respective lists
            lat2, lon2 = coordinates[i]
            lat1, lon1 = coordinates[i-1]
            
            latitudes.append(lat2)
            longitudes.append(lon2)
            
            # Convert decimal degrees to radians for Haversine formula
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula components
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (math.sin(dlat/2)**2 + 
                 math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
            c = 2 * math.asin(math.sqrt(a))
            
            # Calculate distance (Earth's radius in meters = 6371000)
            distance = c * 6371000
            distances.append(distance)
        
        # Create and return DataFrame
        return pd.DataFrame({
            'latitude': latitudes,
            'longitude': longitudes,
            'distance': distances
        })
    
    except Exception as e:
        raise ValueError(f"Error processing polyline: {str(e)}")



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    # Step 2: Transform the rotated matrix
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the current row and column in the rotated matrix, excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix



def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Convert timestamps to datetime
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    # Initialize an empty Series to store results
    results = pd.Series(dtype=bool)

    # Check completeness for each group
    for (id_value, id_2_value), group in grouped:
        full_day = group['start_timestamp'].dt.date.nunique() == 7
        full_time_range = group['start_timestamp'].min() <= group['end_timestamp'].max()
        start_time = group['start_timestamp'].dt.time.min() == pd.to_timedelta(0)
        end_time = group['end_timestamp'].dt.time.max() == pd.to_timedelta('23:59:59')

        results.loc[(id_value, id_2_value)] = full_day and full_time_range and start_time and end_time

    results.index.names = ['id', 'id_2']  # Set multi-index
    return pd.Series()