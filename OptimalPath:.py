from itertools import product
import numpy as np
import requests
import folium
import matplotlib.pyplot as plt
import ripleyReference as rr

def extract_coordinates(data):
    return np.array([[element['lat'], element['lon']] 
                     for element in data['elements'] 
                        if 'lat' in element and 'lon' in element])

def get_square(lat, lon, min_lat, min_lon, lat_step, lon_step):
    lat_idx = int((lat - min_lat) / lat_step)
    lon_idx = int((lon - min_lon) / lon_step)
    return lat_idx, lon_idx

def assign_k_values(data, k_values, k_index, k_matrix, min_lat, min_lon, lat_step, lon_step):
    for (lat, lon), k_value in zip(data, k_values):
        lat_idx, lon_idx = get_square(lat, lon, min_lat, min_lon, lat_step, lon_step)
        k_matrix[lat_idx, lon_idx, k_index] += k_value

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def is_valid_combination(comb, k_matrix, min_lat, min_lon, lat_step, lon_step, percentile):
    threshold = np.percentile(k_matrix.flatten(), percentile)
    seen_blocks = set()
    for place in comb:
        block_index = get_square(place[0], place[1], min_lat, min_lon, lat_step, lon_step)
        if block_index in seen_blocks:
            return False
        if np.any(k_matrix[block_index[0], block_index[1], :] < threshold):
            return False
        seen_blocks.add(block_index)
    return True

def find_shortest_path(place_types, k_matrix, min_lat, min_lon, lat_step, lon_step):
    combinations = product(*place_types)
    min_distance = float('inf')
    best_path = None
    for comb in combinations:
        if is_valid_combination(comb, k_matrix, min_lat, min_lon, lat_step, lon_step, 75):
            distance = sum(calculate_distance(comb[i], comb[i+1]) for i in range(len(comb) - 1))
            if distance < min_distance:
                min_distance = distance
                best_path = comb
    return best_path, min_distance

def create_map(best_path, area):
    m = folium.Map(location=[(area[0] + area[1]) / 2, (area[2] + area[3]) / 2], zoom_start=8)
    folium.PolyLine(best_path, color="blue", weight=2.5, opacity=1).add_to(m)
    for point in best_path:
        folium.Marker(location=point).add_to(m)
    return m

def visualize_k_matrix(k_matrix):
    plt.figure(figsize=(12, 8))
    plt.imshow(np.sum(k_matrix, axis=2), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('K Value Matrix')
    plt.show()

def main(place_types, data_arrays, area, blocks):
    min_lat, max_lat, min_lon, max_lon = area
    num_lat_squares, num_lon_squares = blocks

    lat_step = (max_lat - min_lat) / num_lat_squares
    lon_step = (max_lon - min_lon) / num_lon_squares

    k_matrix = np.zeros((num_lat_squares, num_lon_squares, len(data_arrays)))

    k_values_list = [rr.ripley_k(data, rr.r) for data in data_arrays]

    for i, (data, k_values) in enumerate(zip(data_arrays, k_values_list)):
        assign_k_values(data, k_values, i, k_matrix, min_lat, min_lon, lat_step, lon_step)

    best_path, min_distance = find_shortest_path(place_types, k_matrix, min_lat, min_lon, lat_step, lon_step)
    return best_path, min_distance, k_matrix

if __name__ == "__main__":
    '''
    # Example
    overpass_query_library = """
    [out:json];
    area["ISO3166-2"="US-CA"];
    (
      node["amenity"="library"]["name"~"High library"](area);
      way["amenity"="library"]["name"~"High library"](area);
      rel["amenity"="library"]["name"~"High library"](area);
    );
    out center;
    """
    response_library = requests.get(rr.overpass_url, params={'data': overpass_query_library})
    data_library = extract_coordinates(response_library.json())

    overpass_query_basketball = """
    [out:json];
    area["ISO3166-2"="US-CA"];
    (
      node["leisure"="pitch"]["sport"="basketball"](area);
      way["leisure"="pitch"]["sport"="basketball"](area);
      rel["leisure"="pitch"]["sport"="basketball"](area);
    );
    out center;
    """
    response_basketball = requests.get(rr.overpass_url, params={'data': overpass_query_basketball})
    data_basketball = extract_coordinates(response_basketball.json())

    place_types = [rr.data_mall_1, data_library, rr.data_fitting_1, data_basketball]
    data_arrays = [rr.data_mall_1, data_library, rr.data_fitting_1, data_basketball]
    area = [32.5, 42.0, -124.5, -114.0]
    blocks = [30, 60]
    '''

    best_path, min_distance, k_matrix = main(place_types, data_arrays, area, blocks)
    
    print(f"Best path: {best_path}, Minimum distance: {min_distance}")
    
    map_ = create_map(best_path, area)
    map_.save("best_path_map.html")
    
    visualize_k_matrix(k_matrix)
