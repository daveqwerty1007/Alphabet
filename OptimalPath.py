import numpy as np
import requests
import itertools
import folium

def fetch_location_data(location_type, area="California"):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    area["ISO3166-2"="US-CA"];
    (
      node["{location_type}"](area);
      way["{location_type}"](area);
      rel["{location_type}"](area);
    );
    out center;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    return response.json()['elements']

def ripley_k(data, r):
    n = len(data)
    k = np.zeros(len(r))
    for i in range(n):
        for j, radius in enumerate(r):
            in_circle = np.linalg.norm(data[i] - data, axis=1) <= radius
            k[j] += np.sum(in_circle) - 1  # Subtract 1 to exclude the point itself
    return k / n

def calculate_k_values(locations, grid_size=(30, 60), radius_range=np.linspace(0, 35, 100)):
    all_lats = [loc['lat'] for sublist in locations for loc in sublist]
    all_lons = [loc['lon'] for sublist in locations for loc in sublist]
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    lat_step = (max_lat - min_lat) / grid_size[0]
    lon_step = (max_lon - min_lon) / grid_size[1]

    k_matrix = np.zeros((*grid_size, len(locations), len(radius_range)))

    for k, func_locations in enumerate(locations):
        data = np.array([(loc['lat'], loc['lon']) for loc in func_locations])
        for loc in func_locations:
            lat, lon = loc['lat'], loc['lon']
            grid_x = int((lat - min_lat) / lat_step)
            grid_y = int((lon - min_lon) / lon_step)
            grid_x = min(grid_x, grid_size[0] - 1)
            grid_y = min(grid_y, grid_size[1] - 1)
            if data.shape[0] > 1:
                k_values = ripley_k(data, radius_range)
                k_matrix[grid_x, grid_y, k, :] = k_values

    return k_matrix

def determine_grid_bounds_and_steps(locations):
    all_lats = [loc['lat'] for sublist in locations for loc in sublist]
    all_lons = [loc['lon'] for sublist in locations for loc in sublist]
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    lat_step = (max_lat - min_lat) / 30
    lon_step = (max_lon - min_lon) / 60
    return min_lat, max_lat, min_lon, max_lon, lat_step, lon_step

def find_optimal_path(locations, k_matrix, threshold):
    combinations = list(itertools.product(*locations))
    min_distance = float('inf')
    best_path = None

    def calculate_distance(loc1, loc2):
        return np.linalg.norm(np.array([loc1['lat'], loc1['lon']]) - np.array([loc2['lat'], loc2['lon']]))

    for comb in combinations:
        valid = True
        total_distance = 0
        for i in range(len(comb) - 1):
            distance = calculate_distance(comb[i], comb[i + 1])
            total_distance += distance
            if distance < threshold:
                valid = False
                break
        if valid and total_distance < min_distance:
            min_distance = total_distance
            best_path = comb

    return best_path, min_distance

def visualize_path(path):
    if not path:
        print("No path found.")
        return

    map_center = [(path[0]['lat'] + path[-1]['lat']) / 2, (path[0]['lon'] + path[-1]['lon']) / 2]
    m = folium.Map(location=map_center, zoom_start=8)
    
    for loc in path:
        folium.Marker([loc['lat'], loc['lon']], popup=loc.get('tags', {}).get('name', 'Unknown')).add_to(m)
    
    folium.PolyLine([(loc['lat'], loc['lon']) for loc in path], color="blue", weight=2.5, opacity=1).add_to(m)
    
    return m

def main(location_types):
    all_locations = []
    for location_type in location_types:
        locations = fetch_location_data(location_type)
        all_locations.append(locations)
    
    global min_lat, max_lat, min_lon, max_lon, lat_step, lon_step
    min_lat, max_lat, min_lon, max_lon, lat_step, lon_step = determine_grid_bounds_and_steps(all_locations)
    k_matrix = calculate_k_values(all_locations)

    threshold = 0.1 
    best_path, total_distance = find_optimal_path(all_locations, k_matrix, threshold)

    print("Optimal Path:", best_path)
    print("Total Distance:", total_distance)
    print("K Value Matrix:", k_matrix)

    # Visualize the path
    m = visualize_path(best_path)
    return m


if __name__ == "__main__":
    m = main(["shop=mall", "amenity=library", "shop=clothes", "amenity=school"])
    m.save('optimal_path.html')
