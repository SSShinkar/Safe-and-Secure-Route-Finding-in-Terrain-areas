from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import geemap
import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon, Point, LineString
import requests
import folium
import ee
from folium.plugins import Draw, Search
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
import os
import jwt
from datetime import datetime, timedelta
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import logging
from http import HTTPStatus
import osmnx as ox
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache
from cachetools import TTLCache
app = Flask(__name__)

# Set a secret key for JWT token generation
app.config['SECRET_KEY'] = os.urandom(24)

app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/mydatabase"
mongo = PyMongo(app)
users_collection = mongo.db.users
coordinates_collection = mongo.db.coordinates
emergency_data_collection = mongo.db.emergency_data  

# Initialize Bcrypt for password hashing
bcrypt = Bcrypt(app)

# Initialize Earth Engine API
ee.Authenticate()
ee.Initialize(project='ee-shinkarshruti2003')

# Function to generate JWT token
def generate_token(email):
    token = jwt.encode({
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=1)  # Token expires in 1 hour
    }, app.config['SECRET_KEY'], algorithm='HS256')
    return token

# Function to verify JWT token
def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['email']
    except jwt.ExpiredSignatureError:
        return None  # Token has expired
    except jwt.InvalidTokenError:
        return None  # Invalid token

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Password mismatch check
        if password != confirm_password:
            flash("Passwords do not match! Please try again.", "error")
            return redirect(url_for('sign_up'))

        # Check if user already exists
        if users_collection.find_one({"email": email}):
            flash("Email is already registered!", "error")
            return redirect(url_for('sign_up'))

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # User data to insert into the database
        user_data = {
            "email": email,
            "password": hashed_password
        }

        try:
            users_collection.insert_one(user_data)
            flash("User signed up successfully!", "success")
            return redirect(url_for('sign_in'))
        except Exception as e:
            flash(f"An error occurred while signing up: {e}", "error")
            return redirect(url_for('sign_up'))

    return render_template('sign_up.html')

@app.route('/sign_in', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        data = request.get_json()  # Fetch JSON data from the request
        email = data.get('email')
        password = data.get('password')

        # Fetch user from the database
        user = users_collection.find_one({"email": email})
        if user and bcrypt.check_password_hash(user['password'], password):
            # Generate JWT token
            token = generate_token(email)
            
            # Login successful
            return jsonify({'success': True, 'message': 'Login successful!', 'token': token})
        else:
            # Invalid credentials
            return jsonify({'success': False, 'message': 'Invalid credentials!'})

    # If GET request, render the sign-in page
    return render_template('sign_in.html')

@app.route('/check_session', methods=['GET'])
def check_session():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'logged_in': False})
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        email = payload['email']
        user = users_collection.find_one({"email": email})
        if user:
            return jsonify({'logged_in': True, 'email': email})
        return jsonify({'logged_in': False})
    except jwt.InvalidTokenError:
        return jsonify({'logged_in': False})

from flask import jsonify

@app.route('/logout')
def logout():
    """
    Logs out the user by clearing their session token.
    Returns a JSON response indicating success or failure.
    """
    try:
        # In a real application, you might also clear server-side session data here.
        # For now, we assume the token is stored client-side and cleared by JavaScript.

        # Return a success response
        return jsonify({
            'success': True,
            'message': 'Logged out successfully!'
        }), 200  # HTTP 200 OK status
    except Exception as e:
        # Log the error for debugging
        print(f"Error during logout: {e}")
        
        # Return an error response
        return jsonify({
            'success': False,
            'message': 'An error occurred during logout. Please try again.'
        }), 500  # HTTP 500 Internal Server Error status
@app.route('/tool')
def tool():
    token = request.args.get('token')  # Get the token from the query parameter
    if not token or not verify_token(token):  # Verify the token
        return jsonify({'error': 'Unauthorized access!'}), 401
    return render_template('tool.html')

@app.route('/profile', methods=['GET'])
def profile_page():  # Renamed to avoid endpoint conflict
    token = request.args.get('token')
    if not token or not verify_token(token):
        return jsonify({'error': 'Unauthorized access!'}), 401
    return render_template('profile.html')

@app.route('/get_profile', methods=['GET'])
def get_profile():
    token = request.args.get('token')
    if not token or not verify_token(token):
        return jsonify({'error': 'Unauthorized access!'}), 401
    
    email = verify_token(token)
    user = users_collection.find_one({"email": email})
    
    if not user or not user.get('profile_complete', False):
        return jsonify({
            'email': email,
            'first_name': '',
            'last_name': '',
            'gender': '',
            'birthdate': '',
            'education': '',
            'profile_complete': False
        })
    
    return jsonify({
        'email': user.get('email', email),
        'first_name': user.get('first_name', ''),
        'last_name': user.get('last_name', ''),
        'gender': user.get('gender', ''),
        'birthdate': user.get('birthdate', ''),
        'education': user.get('education', ''),
        'profile_complete': True
    })

@app.route('/update_profile', methods=['POST'])
def update_profile():
    token = request.headers.get('Authorization')
    if not token or not verify_token(token):
        return jsonify({'error': 'Unauthorized access!'}), 401
    
    email = verify_token(token)  # Use token-derived email
    data = request.get_json()
    first_name = data.get('firstName', '')
    last_name = data.get('lastName', '')
    gender = data.get('gender', '')
    birthdate = data.get('birthdate', '')
    education = data.get('education', '')
    # Email from form is ignored; use token-derived email
    profile_complete = all([first_name.strip(), last_name.strip(), gender.strip(), birthdate.strip(), education.strip()])
    
    users_collection.update_one(
        {"email": email},
        {"$set": {
            "email": email,  # Always set to token-derived email
            "first_name": first_name,
            "last_name": last_name,
            "gender": gender,
            "birthdate": birthdate,
            "education": education,
            "profile_complete": profile_complete
        }},
        upsert=True
    )
    return jsonify({'success': True, 'message': 'Profile updated successfully!', 'profile_complete': profile_complete})

@app.route('/generate-map', methods=['POST'])
def generate_map():
    try:
        token = request.headers.get('Authorization')
        if not token or not verify_token(token):
            return jsonify({'error': 'Unauthorized access!'}), 401

        data = request.json

        # Extract user-provided coordinates
        user_coords = [
            [float(data['longitude1']), float(data['latitude1'])],
            [float(data['longitude2']), float(data['latitude2'])],
            [float(data['longitude3']), float(data['latitude3'])],
            [float(data['longitude4']), float(data['latitude4'])]
        ]

        # Create a bounding box from the user-provided coordinates
        user_bounds = ee.Geometry.Polygon([user_coords])

        # Initialize the folium map centered at the average coordinates with high-resolution tiles
        center_lat = sum(coord[1] for coord in user_coords) / 4
        center_lon = sum(coord[0] for coord in user_coords) / 4
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",  # Google Satellite tiles
            attr="Google Satellite"
        )

        # Process each coordinate point
        searchable_points = folium.FeatureGroup(name='Searchable Points').add_to(m)

        for lon, lat in user_coords:
            point = ee.Geometry.Point([lon, lat])

            # Fetch Sentinel-2 imagery for the point
            image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                       .filterBounds(point) \
                       .filterDate('2024-01-01', '2025-02-15') \
                       .sort('CLOUDY_PIXEL_PERCENTAGE') \
                       .first()

            vis_params = {
                'min': 0,
                'max': 3000,
                'bands': ['B4', 'B3', 'B2']
            }

            tile_layer = geemap.ee_tile_layer(image, vis_params, f'Sentinel-2 ({lat}, {lon})')

            folium.TileLayer(
                tiles=tile_layer.url_format,
                attr='Google Earth Engine',
                overlay=True,
                name=f'Sentinel-2 ({lat}, {lon})'
            ).add_to(m)

            folium.Marker(location=[lat, lon], popup=f"Location: ({lat}, {lon})").add_to(searchable_points)

        # Add map controls
        folium.LayerControl().add_to(m)
        Search(layer=searchable_points, geom_type='Point', placeholder="Search for locations").add_to(m)
        Draw(export=True).add_to(m)

        # Insert coordinate into MongoDB
        latest_entry = coordinates_collection.find_one({}, sort=[("Location_ID", -1)])
        new_location_id = latest_entry["Location_ID"] + 1 if latest_entry else 1
        new_state_id = round(latest_entry["State_ID"] + 0.1, 1) if latest_entry else 1.1

        Coordinate_data = {
            "Location_ID": new_location_id,
            "Place_Name": "User Defined Region",
            "Lat_N": max(coord[1] for coord in user_coords),
            "Lat_S": min(coord[1] for coord in user_coords),
            "Lon_E": max(coord[0] for coord in user_coords),
            "Lon_W": min(coord[0] for coord in user_coords),
            "State_ID": new_state_id
        }
        
        result = coordinates_collection.insert_one(Coordinate_data)

        if result.inserted_id:
            print(f"Inserted document ID: {result.inserted_id}")

        # Save the generated map
        map_file = 'static/user_map.html'
        m.save(map_file)

        return jsonify({'map_url': map_file})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/description')
def description():
    return render_template('description1.html')

@app.route('/pathdetect')
def pathdetect():
    return render_template('pathdetect.html')

# Step 2: Fetch data using Overpass API
def fetch_osm_data(bbox, query):
    overpass_url = "http://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={"data": query})
    return response.json()

# Step 3: Create a Folium Map with High-Resolution Satellite Imagery
def create_map(center_location, zoom_start=12):
    return folium.Map(
        location=center_location,
        zoom_start=zoom_start,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",  # Google Satellite tiles
        attr="Google Satellite"  # Attribution for Google tiles
    )

# Step 4: Add forests to the map
def add_forests(map_obj, gdf_forests):
    for idx, row in gdf_forests.iterrows():
        folium.GeoJson(
            row["geometry"],
            style_function=lambda x: {"fillColor": "green", "color": "green", "weight": 2, "fillOpacity": 0.5},
            tooltip="Forest"
        ).add_to(map_obj)

# Step 5: Add water bodies to the map
def add_water_bodies(map_obj, gdf_water_bodies):
    for idx, row in gdf_water_bodies.iterrows():
        folium.GeoJson(
            row["geometry"],
            style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 2, "fillOpacity": 0.5},
            tooltip="Water Body"
        ).add_to(map_obj)

# Step 6: Add peaks to the map
def add_peaks(map_obj, gdf_peaks):
    for idx, row in gdf_peaks.iterrows():
        folium.Marker(
            location=[row["geometry"].y, row["geometry"].x],
            popup="Peak",
            icon=folium.Icon(color="gray", icon="info-sign")
        ).add_to(map_obj)

# Step 7: Add roads to the map
def add_roads(map_obj, gdf_roads):
    for idx, row in gdf_roads.iterrows():
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in row["geometry"].coords],
            color="red",
            weight=3,
            opacity=0.8,
            tooltip="Road"
        ).add_to(map_obj)

# Step 8: Add trails to the map
def add_trails(map_obj, gdf_trails):
    for idx, row in gdf_trails.iterrows():
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in row["geometry"].coords],
            color="black",
            weight=2,
            opacity=0.8,
            tooltip="Trail"
        ).add_to(map_obj)

# Step 9: Add waterways to the map
def add_waterways(map_obj, gdf_waterways):
    for idx, row in gdf_waterways.iterrows():
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in row["geometry"].coords],
            color="darkblue",
            weight=2,
            opacity=0.8,
            tooltip="Waterway"
        ).add_to(map_obj)

# Step 10: Add markers for specific places
def add_places(map_obj, places):
    for place, coords in places.items():
        folium.Marker(
            location=coords,
            popup=place,
            icon=folium.Icon(color="orange", icon="tree-deciduous")
        ).add_to(map_obj)

@app.route('/process-routes', methods=['POST'])
def process_routes():
    try:
        data = request.json
        user_coords = [
            [float(data['longitude1']), float(data['latitude1'])],
            [float(data['longitude2']), float(data['latitude2'])],
            [float(data['longitude3']), float(data['latitude3'])],
            [float(data['longitude4']), float(data['latitude4'])]
        ]

        # Create a bounding box from the user-provided coordinates
        south = min(coord[1] for coord in user_coords)
        north = max(coord[1] for coord in user_coords)
        west = min(coord[0] for coord in user_coords)
        east = max(coord[0] for coord in user_coords)

        # Define the Overpass query
        query = f"""
        [out:json];
        (
          // Fetch natural features like forests, hills, and water bodies
          way["natural"="wood"](bbox:{south},{west},{north},{east});
          way["natural"="water"](bbox:{south},{west},{north},{east});
          way["natural"="peak"](bbox:{south},{west},{north},{east});
          relation["natural"="wood"](bbox:{south},{west},{north},{east});
          relation["natural"="water"](bbox:{south},{west},{north},{east});
          relation["natural"="peak"](bbox:{south},{west},{north},{east});

          // Fetch roads and paths
          way["highway"](bbox:{south},{west},{north},{east});

          // Fetch trails (footways, paths, etc.)
          way["highway"="footway"](bbox:{south},{west},{north},{east});
          way["highway"="path"](bbox:{south},{west},{north},{east});

          // Fetch waterways (rivers, streams, etc.)
          way["waterway"](bbox:{south},{west},{north},{east});
        );
        out geom;
        """

        response = requests.get("http://overpass-api.de/api/interpreter", params={"data": query})
        data = response.json()

        # Convert OSM data to GeoDataFrame
        roads = []
        trails = []
        forests = []
        water_bodies = []
        waterways = []

        for element in data["elements"]:
            if element["type"] == "way" and "geometry" in element:
                coords = [(node["lon"], node["lat"]) for node in element["geometry"]]
                if element.get("tags", {}).get("highway") in ["footway", "path"]:
                    trails.append({"geometry": LineString(coords), "type": "trail"})
                elif element.get("tags", {}).get("highway"):
                    roads.append({"geometry": LineString(coords), "type": "road"})
                elif element.get("tags", {}).get("natural") == "wood":
                    forests.append({"geometry": Polygon(coords), "type": "forest"})
                elif element.get("tags", {}).get("natural") == "water":
                    water_bodies.append({"geometry": Polygon(coords), "type": "water"})
                elif element.get("tags", {}).get("waterway"):
                    waterways.append({"geometry": LineString(coords), "type": "waterway"})

        gdf_roads = gpd.GeoDataFrame(roads, crs="EPSG:4326")
        gdf_trails = gpd.GeoDataFrame(trails, crs="EPSG:4326")
        gdf_forests = gpd.GeoDataFrame(forests, crs="EPSG:4326")
        gdf_water_bodies = gpd.GeoDataFrame(water_bodies, crs="EPSG:4326")
        gdf_waterways = gpd.GeoDataFrame(waterways, crs="EPSG:4326")

        # Create a new map with high-resolution satellite imagery
        center_lat = sum(coord[1] for coord in user_coords) / 4
        center_lon = sum(coord[0] for coord in user_coords) / 4
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",  # Google Satellite tiles
            attr="Google Satellite"
        )

        # Add layers to the map
        def add_layer(map_obj, gdf, color, name):
            if not gdf.empty:
                folium.GeoJson(
                    gdf,
                    style_function=lambda x: {"color": color, "weight": 2, "opacity": 0.8},
                    name=name
                ).add_to(map_obj)

        add_layer(m, gdf_roads, "red", "Roads")
        add_layer(m, gdf_trails, "black", "Trails")
        add_layer(m, gdf_forests, "green", "Forests")
        add_layer(m, gdf_water_bodies, "blue", "Water Bodies")
        add_layer(m, gdf_waterways, "darkblue", "Waterways")

        # Add layer control
        folium.LayerControl().add_to(m)

        # Add JavaScript to handle layer toggling
        map_html = m.get_root().render()
        map_html += """
        <script>
        // Function to toggle layer visibility
        function toggleLayer(layer) {
            const layers = {
                roads: document.querySelector('.leaflet-layer:has(.leaflet-geojson-layer[fill="red"])'),
                trails: document.querySelector('.leaflet-layer:has(.leaflet-geojson-layer[fill="black"])'),
                forests: document.querySelector('.leaflet-layer:has(.leaflet-geojson-layer[fill="green"])'),
                water_bodies: document.querySelector('.leaflet-layer:has(.leaflet-geojson-layer[fill="blue"])'),
                waterways: document.querySelector('.leaflet-layer:has(.leaflet-geojson-layer[fill="darkblue"])')
            };

            // Hide all layers first
            Object.values(layers).forEach(layer => {
                if (layer) layer.style.display = 'none';
            });

            // Show the selected layer
            if (layers[layer]) {
                layers[layer].style.display = 'block';
            }
        }

        // Listen for messages from the parent window
        window.addEventListener('message', (event) => {
            if (event.data.type === 'toggleLayer') {
                toggleLayer(event.data.layer);
            }
        });
        </script>
        """

        # Save the map
        map_file = 'static/processed_map.html'
        with open(map_file, 'w') as f:
            f.write(map_html)

        return jsonify({'map_url': map_file})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/userselect')
def userselect():
    token = request.args.get('token')  # Get the token from the query parameter
    if not token or not verify_token(token):  # Verify the token
        return jsonify({'error': 'Unauthorized access!'}), 401
    return render_template('userselect.html')

@app.route('/optimalroute')
def optimalroute():
    token = request.args.get('token')  # Get the token from the query parameter
    if not token or not verify_token(token):  # Verify the token
        return jsonify({'error': 'Unauthorized access!'}), 401
    return render_template('optimalroute.html')

@app.route('/merge-and-store-emergency-data', methods=['POST'])
def merge_and_store_emergency_data():
    try:
        print("Starting to merge and store emergency data...")

        # Read Excel files
        earthquake_data = pd.read_excel('./static/Earthquack_data.xlsx')
        landslide_data = pd.read_excel('./static/Landslides and rockfalls area in india.xlsx')
        flood_data = pd.read_excel('./static/flash flood prone place.xlsx')

        # Add a 'type' field to distinguish between datasets
        earthquake_data['type'] = 'earthquake'
        landslide_data['type'] = 'landslide'
        flood_data['type'] = 'flood'

        # Combine all data into a single DataFrame
        combined_data = pd.concat([earthquake_data, landslide_data, flood_data], ignore_index=True)

        # Convert the combined DataFrame to a list of dictionaries
        records = combined_data.to_dict('records')

        # Check if the collection exists, if not, create it
        if 'emergency_data' not in mongo.db.list_collection_names():
            print("Creating 'emergency_data' collection...")
            mongo.db.create_collection('emergency_data')

        # Check if the collection is empty
        if emergency_data_collection.count_documents({}) == 0:
            # Insert records into the collection
            emergency_data_collection.insert_many(records)
            print("Emergency data inserted into MongoDB successfully.")
            return jsonify({'success': True, 'message': 'Emergency data merged and stored successfully!'})
        else:
            print("Emergency data already exists in the database. Skipping insertion.")
            return jsonify({'success': True, 'message': 'Emergency data already exists in the database.'})

    except FileNotFoundError as e:
        error_message = f"Excel file not found: {str(e)}"
        print(error_message)
        return jsonify({'success': False, 'message': error_message}), 404

    except pd.errors.EmptyDataError as e:
        error_message = f"Excel file is empty: {str(e)}"
        print(error_message)
        return jsonify({'success': False, 'message': error_message}), 400

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return jsonify({'success': False, 'message': error_message}), 500


@app.route('/get-emergency-data', methods=['POST'])
def get_emergency_data():
    try:
        data = request.json
        emergency_type = data.get('type')
        if emergency_type not in ['earthquake', 'landslide', 'flood']:
            return jsonify({'error': 'Invalid emergency type'}), 400

        # Get the selected region coordinates from the request
        user_coords = [
            [float(data['longitude1']), float(data['latitude1'])],
            [float(data['longitude2']), float(data['latitude2'])],
            [float(data['longitude3']), float(data['latitude3'])],
            [float(data['longitude4']), float(data['latitude4'])]
        ]

        # Create a bounding box from the user-provided coordinates
        south = min(coord[1] for coord in user_coords)
        north = max(coord[1] for coord in user_coords)
        west = min(coord[0] for coord in user_coords)
        east = max(coord[0] for coord in user_coords)

        # Fetch data from MongoDB based on the type
        data = list(emergency_data_collection.find({"type": emergency_type}))

        # Prepare source and destination data
        source_all = []
        destination_all = []
        source_in_region = []
        destination_in_region = []

        for record in data:
            # Add source city data (all cities)
            source_all.append({
                "name": record["City"],
                "lat": record["Latitude"],
                "lon": record["Longitude"]
            })

            # Add destination city data (all safer nearby cities)
            destination_all.append({
                "name": record["Safer Nearby City"],
                "lat": record["Safe Latitude"],
                "lon": record["Safe Longitude"]
            })

            # Check if the source city is within the selected region
            if (south <= record["Latitude"] <= north) and (west <= record["Longitude"] <= east):
                source_in_region.append({
                    "name": record["City"],
                    "lat": record["Latitude"],
                    "lon": record["Longitude"],
                    # Include safer nearby city details for this source city
                    "SaferNearbyCity": record["Safer Nearby City"],
                    "SafeLatitude": record["Safe Latitude"],
                    "SafeLongitude": record["Safe Longitude"]
                })

            # Check if the safer nearby city is within the selected region
            if (south <= record["Safe Latitude"] <= north) and (west <= record["Safe Longitude"] <= east):
                destination_in_region.append({
                    "name": record["Safer Nearby City"],
                    "lat": record["Safe Latitude"],
                    "lon": record["Safe Longitude"]
                })

        return jsonify({
            'source_all': source_all,
            'destination_all': destination_all,
            'source_in_region': source_in_region,
            'destination_in_region': destination_in_region
        })

    except Exception as e:
        print(f"Error in /get-emergency-data: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500
@app.route('/get-cities-under-coordinates', methods=['POST'])
def get_cities_under_coordinates():
    try:
        data = request.json 
        user_coords = [
            [float(data['longitude1']), float(data['latitude1'])],
            [float(data['longitude2']), float(data['latitude2'])],
            [float(data['longitude3']), float(data['latitude3'])],
            [float(data['longitude4']), float(data['latitude4'])]
        ]

        # Create a bounding box from the user-provided coordinates
        south = min(coord[1] for coord in user_coords)
        north = max(coord[1] for coord in user_coords)
        west = min(coord[0] for coord in user_coords)
        east = max(coord[0] for coord in user_coords)

        # Query the emergency_data collection for cities within the bounding box
        cities = earthquick_collection.find({
            "Latitude": {"$gte": south, "$lte": north},
            "Longitude": {"$gte": west, "$lte": east}
        })

        # Convert the cursor to a list of dictionaries
        cities_list = []
        for city in cities:
            cities_list.append({
                "name": city.get("City"),
                "lat": city.get("Latitude"),
                "lon": city.get("Longitude")
            })

        return jsonify({'cities': cities_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Configure logging for debugging

# Configure logging for debugging


@app.route('/get-places', methods=['POST'])
def get_places():
    try:
        data = request.json
        purpose = data.get('purpose')
        user_coords = [
            [float(data['longitude1']), float(data['latitude1'])],
            [float(data['longitude2']), float(data['latitude2'])],
            [float(data['longitude3']), float(data['latitude3'])],
            [float(data['longitude4']), float(data['latitude4'])]
        ]

        # Create a bounding box from the user-provided coordinates
        south = min(coord[1] for coord in user_coords)
        north = max(coord[1] for coord in user_coords)
        west = min(coord[0] for coord in user_coords)
        east = max(coord[0] for coord in user_coords)
        logging.info(f"Bounding box: S:{south}, N:{north}, W:{west}, E:{east}")

        if purpose == 'Adventure & Trekking':
            try:
                # Check if openpyxl is installed
                try:
                    import openpyxl
                except ImportError:
                    error_msg = "The 'openpyxl' library is not installed. Please install it using 'pip install openpyxl'."
                    logging.error(error_msg)
                    return jsonify({'error': error_msg}), 500

                # Define Excel files for Adventure & Trekking categories
                excel_files = {
                    "Tourism Place": './static/Tourisms.xlsx',
                    "Trekking Place": './static/Treeking.xlsx',
                    "Adventure Place": './static/Adventure.xlsx',
                    "Desert Place": './static/Desert.xlsx',
                    "Waterbodies": './static/Waterbodies.xlsx'
                }

                # Initialize categories
                categories = {
                    "Tourism Place": [],
                    "Trekking Place": [],
                    "Adventure Place": [],
                    "Desert Place": [],
                    "Waterbodies": [],
                    "City": []
                }

                # Helper function to parse mixed coordinate formats
                def parse_coordinate(coord):
                    if pd.isna(coord):
                        return None
                    try:
                        coord_str = str(coord).strip()
                        if '° S' in coord_str:
                            return -float(coord_str.replace('° S', '').strip())
                        elif '° W' in coord_str:
                            return -float(coord_str.replace('° W', '').strip())
                        elif '° N' in coord_str or '° E' in coord_str:
                            return float(coord_str.replace('° N', '').replace('° E', '').strip())
                        else:
                            return float(coord_str)
                    except (ValueError, TypeError) as e:
                        logging.debug(f"Failed to parse coordinate '{coord}': {str(e)}")
                        return None

                # Process each Excel file with category-specific column names
                for category, excel_path in excel_files.items():
                    if not os.path.exists(excel_path):
                        logging.warning(f"File not found: {excel_path}")
                        continue

                    if not os.access(excel_path, os.R_OK):
                        error_msg = f"Permission denied: Cannot read {excel_path}"
                        logging.error(error_msg)
                        return jsonify({'error': error_msg}), 403

                    try:
                        # Load Excel file
                        adventure_data = pd.read_excel(excel_path)
                        logging.info(f"Loaded {excel_path} with {len(adventure_data)} rows")
                        columns = adventure_data.columns.tolist()
                        logging.debug(f"Columns in {excel_path}: {columns}")

                        # Define expected column names based on category
                        if category == "Tourism Place":
                            expected_name = "tourism place"
                            expected_lat = "tourism place latitude"
                            expected_lon = "tourism place longitude"
                        elif category == "Trekking Place":
                            expected_name = "trekking place"
                            expected_lat = "trekking place latitude"
                            expected_lon = "trekking place longitude"
                        elif category == "Adventure Place":
                            expected_name = "adventure place"
                            expected_lat = "adventure place latitude"
                            expected_lon = "adventure place  longitude"
                        elif category == "Desert Place":
                            expected_name = "desert place"
                            expected_lat = "desert place latitude"
                            expected_lon = "desert place longitude"
                        elif category == "Waterbodies":
                            expected_name = "waterbodies place"
                            expected_lat = "waterbodies latitude"
                            expected_lon = "waterbodies longitude"

                        # Find columns (case-insensitive)
                        name_col = next((col for col in columns if col.lower() == expected_name), None)
                        lat_col = next((col for col in columns if col.lower() == expected_lat), None)
                        lon_col = next((col for col in columns if col.lower() == expected_lon), None)

                        if not all([name_col, lat_col, lon_col]):
                            logging.warning(f"Missing required columns in {excel_path}: name={name_col}, lat={lat_col}, lon={lon_col}")
                            continue

                        places_added = 0
                        for index, row in adventure_data.iterrows():
                            try:
                                name = str(row[name_col]).strip() if pd.notna(row[name_col]) else None
                                lat = parse_coordinate(row[lat_col])
                                lon = parse_coordinate(row[lon_col])

                                if name and lat is not None and lon is not None:
                                    if south <= lat <= north and west <= lon <= east:
                                        categories[category].append({"name": name, "lat": lat, "lon": lon})
                                        places_added += 1
                                    else:
                                        logging.debug(f"Excluded {name} from {category}: ({lat}, {lon}) outside region S:{south}, N:{north}, W:{west}, E:{east}")
                                else:
                                    logging.debug(f"Skipped row {index} in {excel_path}: name={name}, lat={row[lat_col]}, lon={row[lon_col]}")

                            except Exception as e:
                                logging.warning(f"Error processing row {index} in {excel_path}: {row.to_dict()}, Error: {str(e)}")
                                continue

                        logging.info(f"Added {places_added} places to '{category}' from {excel_path}")

                    except pd.errors.EmptyDataError:
                        logging.warning(f"Empty file: {excel_path}")
                        continue
                    except Exception as e:
                        logging.error(f"Error reading {excel_path}: {str(e)}")
                        continue

                # Fetch cities and important places for "City" category via Overpass API
                city_query = f"""
                [out:json];
                (
                  node["place"~"city|town|village"](bbox:{south},{west},{north},{east});
                  way["place"~"city|town|village"](bbox:{south},{west},{north},{east});
                  relation["place"~"city|town|village"](bbox:{south},{west},{north},{east});
                  node["tourism"](bbox:{south},{west},{north},{east});
                  way["tourism"](bbox:{south},{west},{north},{east});
                  node["amenity"](bbox:{south},{west},{north},{east});
                  way["amenity"](bbox:{south},{west},{north},{east});
                );
                out center;
                """

                try:
                    response = requests.get("http://overpass-api.de/api/interpreter", params={"data": city_query}, timeout=30)
                    if response.ok:
                        city_data = response.json()
                        seen_places = set()
                        for element in city_data["elements"]:
                            if "tags" not in element or "name" not in element["tags"]:
                                continue
                            name = element["tags"].get("name", "").strip()
                            if not name or name in seen_places:
                                continue
                            lat = element.get("lat") or element.get("center", {}).get("lat")
                            lon = element.get("lon") or element.get("center", {}).get("lon")
                            if lat and lon:
                                categories["City"].append({"name": name, "lat": float(lat), "lon": float(lon)})
                                seen_places.add(name)
                        logging.info(f"Added {len(categories['City'])} places to 'City' category from Overpass API")
                    else:
                        logging.warning(f"Overpass API failed: {response.status_code} - {response.text}")
                except requests.RequestException as e:
                    logging.warning(f"Overpass API error: {str(e)}")

                # Check and return response
                total_places = sum(len(places) for places in categories.values())
                if total_places == 0:
                    logging.error(f"No places found for Adventure & Trekking in region (S:{south}, N:{north}, W:{west}, E:{east})")
                    return jsonify({
                        'categories': categories,
                        'error': 'No places found in the selected region. Check Excel files and coordinates.'
                    }), 200
                else:
                    logging.info(f"Returning {total_places} places across all categories")
                    return jsonify({'categories': categories})

            except Exception as e:
                logging.error(f"Unexpected error in Adventure & Trekking: {str(e)}")
                return jsonify({'error': f"Server error: {str(e)}"}), 500

        # Unchanged logic for Transport & Logistics
        elif purpose == 'Transport & Logistics':
            query = f"""
            [out:json];
            (
              node["industrial"](bbox:{south},{west},{north},{east});
              way["industrial"](bbox:{south},{west},{north},{east});
              node["building"="warehouse"](bbox:{south},{west},{north},{east});
              way["building"="warehouse"](bbox:{south},{west},{north},{east});
              node["landuse"="farmland"](bbox:{south},{west},{north},{east});
              way["landuse"="farmland"](bbox:{south},{west},{north},{east});
              node["harbour"](bbox:{south},{west},{north},{east});
              way["harbour"](bbox:{south},{west},{north},{east});
              node["aeroway"="aerodrome"](bbox:{south},{west},{north},{east});
              way["aeroway"="aerodrome"](bbox:{south},{west},{north},{east});
              node["railway"="station"](bbox:{south},{west},{north},{east});
              way["railway"="station"](bbox:{south},{west},{north},{east});
              node["building"="logistics"](bbox:{south},{west},{north},{east});
              way["building"="logistics"](bbox:{south},{west},{north},{east});
              node["place"~"city|town|village"](bbox:{south},{west},{north},{east});
              way["place"~"city|town|village"](bbox:{south},{west},{north},{east});
              relation["place"~"city|town|village"](bbox:{south},{west},{north},{east});
              node["shop"](bbox:{south},{west},{north},{east});
              way["shop"](bbox:{south},{west},{north},{east});
              node["building"="residential"](bbox:{south},{west},{north},{east});
              way["building"="residential"](bbox:{south},{west},{north},{east});
              node["building"="commercial"](bbox:{south},{west},{north},{east});
              way["building"="commercial"](bbox:{south},{west},{north},{east});
            );
            out center;
            """

            response = requests.get("http://overpass-api.de/api/interpreter", params={"data": query})
            if not response.ok:
                return jsonify({'error': 'Failed to fetch data from Overpass API'}), 500
            data = response.json()

            source_categories = {
                "Factories & Manufacturing Product Place": [],
                "Warehouse": [],
                "Farms & Agricultural Fields": [],
                "Ports & Docks": [],
                "Airports": [],
                "Railway Stations": [],
                "Distribution Centers": [],
                "City": []
            }
            destination_categories = {
                "Retail Source & Markets": [],
                "Warehouse & Distribution Center": [],
                "Factories & Processing Units": [],
                "Homes & Business": [],
                "City": []
            }

            seen_places = set()
            for element in data["elements"]:
                if "tags" not in element or "name" not in element["tags"]:
                    continue

                name = element["tags"]["name"]
                if not name or name in seen_places or not name.strip():
                    continue

                lat = element.get("lat") or element.get("center", {}).get("lat")
                lon = element.get("lon") or element.get("center", {}).get("lon")
                if not (lat and lon):
                    continue

                place = {"name": name, "lat": lat, "lon": lon}
                tags = element["tags"]
                seen_places.add(name)

                if "industrial" in tags:
                    source_categories["Factories & Manufacturing Product Place"].append(place)
                    destination_categories["Factories & Processing Units"].append(place)
                elif tags.get("building") == "warehouse":
                    source_categories["Warehouse"].append(place)
                    destination_categories["Warehouse & Distribution Center"].append(place)
                elif tags.get("landuse") == "farmland":
                    source_categories["Farms & Agricultural Fields"].append(place)
                elif "harbour" in tags:
                    source_categories["Ports & Docks"].append(place)
                elif tags.get("aeroway") == "aerodrome":
                    source_categories["Airports"].append(place)
                elif tags.get("railway") == "station":
                    source_categories["Railway Stations"].append(place)
                elif tags.get("building") == "logistics":
                    source_categories["Distribution Centers"].append(place)
                    destination_categories["Warehouse & Distribution Center"].append(place)
                elif tags.get("place") in ["city", "town", "village"]:
                    source_categories["City"].append(place)
                    destination_categories["City"].append(place)
                elif "shop" in tags:
                    destination_categories["Retail Source & Markets"].append(place)
                elif tags.get("building") in ["residential", "commercial"]:
                    destination_categories["Homes & Business"].append(place)

            return jsonify({
                'source_categories': source_categories,
                'destination_categories': destination_categories
            })

        # Unchanged logic for General Use
        elif purpose == 'General Use':
            query = f"""
            [out:json];
            (
              node["place"](bbox:{south},{west},{north},{east});
              way["place"](bbox:{south},{west},{north},{east});
              relation["place"](bbox:{south},{west},{north},{east});
              node["natural"="wood"](bbox:{south},{west},{north},{east});
              way["natural"="wood"](bbox:{south},{west},{north},{east});
              relation["natural"="wood"](bbox:{south},{west},{north},{east});
              node["waterway"](bbox:{south},{west},{north},{east});
              way["waterway"](bbox:{south},{west},{north},{east});
              relation["waterway"](bbox:{south},{west},{north},{east});
              node["tourism"](bbox:{south},{west},{north},{east});
              way["tourism"](bbox:{south},{west},{north},{east});
              relation["tourism"](bbox:{south},{west},{north},{east});
              node["amenity"](bbox:{south},{west},{north},{east});
              way["amenity"](bbox:{south},{west},{north},{east});
              relation["amenity"](bbox:{south},{west},{north},{east});
            );
            out center;
            """

            import time
            max_retries = 3
            retry_delay = 10
            timeout = 60

            for attempt in range(max_retries):
                try:
                    response = requests.get("http://overpass-api.de/api/interpreter", params={"data": query}, timeout=timeout)
                    if not response.ok:
                        logging.error(f"Overpass API failed with status {response.status_code}: {response.text}")
                        return jsonify({'error': f'Failed to fetch data from Overpass API: {response.text}'}), 500
                    
                    data = response.json()
                    if not data.get("elements"):
                        logging.info(f"No elements found in Overpass response for bbox: {south},{west},{north},{east}")
                        return jsonify({'places': []})

                    places = []
                    seen_places = set()
                    for element in data["elements"]:
                        if "tags" not in element or "name" not in element["tags"]:
                            continue

                        name = element["tags"].get("name", "").strip()
                        if not name or name in seen_places:
                            continue

                        lat = element.get("lat") or element.get("center", {}).get("lat")
                        lon = element.get("lon") or element.get("center", {}).get("lon")
                        if not (lat and lon):
                            continue

                        place_type = element["tags"].get("place", element["tags"].get("natural", element["tags"].get("waterway", element["tags"].get("tourism", element["tags"].get("amenity", "unknown")))))
                        place = {
                            "name": name,
                            "type": place_type,
                            "lat": float(lat),
                            "lon": float(lon)
                        }
                        places.append(place)
                        seen_places.add(name)
                        if place_type in ["city", "town", "village"]:
                            logging.debug(f"Added city/town/village: {name} ({lat}, {lon})")
                        elif place_type in ["tourism", "amenity"]:
                            logging.debug(f"Added important spot: {name} ({lat}, {lon}, type: {place_type})")

                    logging.debug(f"Found {len(places)} places for General Use in bbox: {south},{west},{north},{east}")
                    return jsonify({'places': places})

                except requests.Timeout as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Attempt {attempt + 1} timed out. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        area = (north - south) * (east - west)
                        logging.error(f"All {max_retries} attempts timed out for bbox: {south},{west},{north},{east}. Area: {area:.2f} sq degrees")
                        return jsonify({
                            'error': f'Overpass API request timed out after {max_retries} attempts (total {max_retries * timeout}s). The selected region might be too large (approx. {area:.2f} sq degrees) or the server is overloaded. Try a smaller region or retry later.'
                        }), 500
                except requests.RequestException as e:
                    logging.error(f"Overpass API request failed: {str(e)}")
                    return jsonify({'error': f'Overpass API request failed: {str(e)}'}), 500
                except ValueError as e:
                    logging.error(f"Data parsing error in General Use: {str(e)}")
                    return jsonify({'error': f'Invalid data format: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"Unexpected error in /get-places: {str(e)}")
        return jsonify({'error': str(e)}), 500
# Find optimal route between source and destination
ORS_API_KEY = '5b3ce3597851110001cf624889850cdc5bee4e2aa2c1a0c9df5dcd30'

@app.route('/find-route', methods=['POST'])
def find_route():
    try:
        data = request.json
        if not data or 'source' not in data or 'destination' not in data:
            return jsonify({'error': 'Invalid request: source and destination coordinates are required'}), HTTPStatus.BAD_REQUEST

        source = data['source']
        destination = data['destination']

        # Validate source and destination coordinates
        for coord in (source, destination):
            if not isinstance(coord, dict) or 'lat' not in coord or 'lon' not in coord:
                return jsonify({'error': 'Invalid coordinates: source and destination must contain lat and lon'}), HTTPStatus.BAD_REQUEST

        # Fetch multiple existing routes using OpenRouteService
        profiles = ['driving-car', 'driving-hgv']
        preferences = ['recommended', 'fastest', 'shortest']
        all_routes = []

        for profile in profiles:
            for pref in preferences:
                url = f'https://api.openrouteservice.org/v2/directions/{profile}/geojson'
                headers = {
                    'Authorization': ORS_API_KEY,
                    'Content-Type': 'application/json',
                    'Accept': 'application/json, application/geo+json'
                }
                body = {
                    "coordinates": [[source['lon'], source['lat']], [destination['lon'], destination['lat']]],
                    "preference": pref,
                    "format": "geojson"
                }
                try:
                    response = requests.post(url, json=body, headers=headers)
                    if response.status_code == 200:
                        route_data = response.json()
                        if 'features' in route_data:
                            for feature in route_data['features']:
                                coords = feature['geometry']['coordinates']
                                # Convert [lon, lat] to [lat, lon] for Leaflet
                                route_coords = [[coord[1], coord[0]] for coord in coords]
                                all_routes.append(route_coords)
                    else:
                        print(f"ORS request failed for {profile}/{pref}: {response.status_code} - {response.text}")
                except requests.RequestException as e:
                    print(f"Error fetching route for {profile}/{pref}: {str(e)}")

        if not all_routes:
            return jsonify({'error': 'No valid routes found connecting source and destination'}), HTTPStatus.NOT_FOUND

        # Return the list of route coordinates
        return jsonify({'routes': all_routes})

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), HTTPStatus.INTERNAL_SERVER_ERROR

# Simulated route storage (replace with MongoDB integration if needed)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
stored_routes = []

@app.route('/route-safety-analysis', methods=['GET', 'POST'])
def route_safety_analysis():
    token = request.args.get('token')
    if not token or not verify_token(token):
        return jsonify({'error': 'Unauthorized access!'}), 401

    coords = {
        'lat1': request.args.get('lat1'), 'lon1': request.args.get('lon1'),
        'lat2': request.args.get('lat2'), 'lon2': request.args.get('lon2'),
        'lat3': request.args.get('lat3'), 'lon3': request.args.get('lon3'),
        'lat4': request.args.get('lat4'), 'lon4': request.args.get('lon4'),
        'sourceLat': request.args.get('sourceLat'), 'sourceLon': request.args.get('sourceLon'),
        'destLat': request.args.get('destLat'), 'destLon': request.args.get('destLon')
    }

    if not all(coords.values()):
        return jsonify({'error': 'Missing coordinates!'}), 400

    if request.method == 'POST' and 'routes' in request.json:
        global stored_routes
        stored_routes = request.json['routes']
        logger.info('Stored routes updated from frontend: %s', stored_routes)

    return render_template('route_safety_analysis.html', coords=coords)

@app.route('/get-routes-for-analysis', methods=['POST'])
def get_routes_for_analysis():
    start_time = time.time()
    try:
        token = request.headers.get('Authorization')
        if not token or not verify_token(token):
            return jsonify({'error': 'Unauthorized access!'}), 401

        data = request.json
        if not data:
            return jsonify({'error': 'No data provided in request'}), 400

        user_coords = [
            [float(data['longitude1']), float(data['latitude1'])],
            [float(data['longitude2']), float(data['latitude2'])],
            [float(data['longitude3']), float(data['latitude3'])],
            [float(data['longitude4']), float(data['latitude4'])]
        ]

        south = min(coord[1] for coord in user_coords)
        north = max(coord[1] for coord in user_coords)
        west = min(coord[0] for coord in user_coords)
        east = max(coord[0] for coord in user_coords)

        routes = stored_routes if stored_routes else []
        if not routes:
            query = f"""
            [out:json];
            (way["highway"](bbox:{south},{west},{north},{east});
             way["highway"="footway"](bbox:{south},{west},{north},{east});
             way["highway"="path"](bbox:{south},{west},{north},{east}););
            out geom;
            """
            osm_data = fetch_osm_data([south, west, north, east], query)
            for element in osm_data["elements"]:
                if element["type"] == "way" and "geometry" in element:
                    coords = [(node["lat"], node["lon"]) for node in element["geometry"]]
                    route_name = element.get("tags", {}).get("name", f"Route {len(routes) + 1}")
                    routes.append({"name": route_name, "coords": coords[:5]})  # Limit OSM coords
            logger.info('Fetched OSM routes as fallback: %s', routes)

        if not routes:
            return jsonify({'error': 'No routes found in the selected region'}), 404

        # Batch fetch terrain data
        all_points = [ee.Geometry.Point(route["coords"][0][1], route["coords"][0][0]) for route in routes]
        terrain_data = ee.Image('MODIS/006/MCD12Q1/2020_01_01').select('LC_Type1').reduceRegions(
            collection=ee.FeatureCollection(all_points), reducer=ee.Reducer.first(), scale=500
        ).getInfo()
        terrain_values = {i: feat['properties']['LC_Type1'] for i, feat in enumerate(terrain_data['features'])}
        logger.info("Terrain data fetched in %.2f seconds", time.time() - start_time)

        # Batch fetch weather data
        weather_values = {}
        for i, route in enumerate(routes):
            lat, lon = route["coords"][0][0], route["coords"][0][1]
            weather_values[i] = analyze_weather_impact([(lat, lon)])

        emergency_data = {e_type: get_emergency_data_helper(user_coords, e_type).get('source_in_region', [])
                          for e_type in ['earthquake', 'landslide', 'flood']}
        osm_data = fetch_osm_data([south, west, north, east], query) if 'osm_data' not in locals() else osm_data

        def analyze_route(route_idx):
            route = routes[route_idx]
            coords = route["coords"][:5]  # Limit to 5 points
            elevation = get_elevation(coords)
            safety = {
                "elevationSlope": analyze_elevation_slope(elevation),
                "terrainType": terrain_values.get(route_idx, 0) not in [1, 2, 3, 4, 5, 13],
                "weatherImpact": weather_values.get(route_idx, True),
                "obstacles": analyze_obstacles(coords, osm_data),
                "landslideRisk": analyze_emergency_risk(coords, emergency_data.get('landslide', [])),
                "wildlifeThreats": analyze_wildlife_threats(coords)
            }
            return {"name": route["name"], "coords": coords, "safety": safety}

        with ThreadPoolExecutor(max_workers=4) as executor:
            future = executor.map(analyze_route, range(len(routes)))
            analyzed_routes = []
            for result in future:
                if time.time() - start_time > 180:  # 3-minute limit
                    logger.warning("Analysis exceeded 180 seconds, returning partial results")
                    return jsonify({'routes': analyzed_routes, 'warning': 'Analysis timed out, results may be incomplete'})
                analyzed_routes.append(result)

        logger.info("Total analysis completed in %.2f seconds", time.time() - start_time)
        return jsonify({'routes': analyzed_routes})

    except ValueError as e:
        logger.error('Invalid coordinate data: %s', str(e))
        return jsonify({'error': f'Invalid coordinate data: {str(e)}'}), 400
    except Exception as e:
        logger.error('Error in get_routes_for_analysis: %s', str(e))
        return jsonify({'error': f'Error analyzing routes: {str(e)}'}), 500

# Helper and analysis functions (unchanged except for minor optimizations)
def get_emergency_data_helper(user_coords, emergency_type):
    south, north = min(coord[1] for coord in user_coords), max(coord[1] for coord in user_coords)
    west, east = min(coord[0] for coord in user_coords), max(coord[0] for coord in user_coords)
    data = list(emergency_data_collection.find({"type": emergency_type}))
    return {'source_in_region': [{"name": r["City"], "lat": r["Latitude"], "lon": r["Longitude"]}
                                for r in data if south <= r["Latitude"] <= north and west <= r["Longitude"] <= east]}

def get_elevation(coords):
    points = [ee.Geometry.Point(lon, lat) for lat, lon in coords]
    elev_data = ee.Image('USGS/SRTMGL1_003').reduceRegions(
        collection=ee.FeatureCollection(points), reducer=ee.Reducer.mean(), scale=30
    ).getInfo()
    return [feat['properties']['mean'] for feat in elev_data['features']]

def analyze_elevation_slope(elevation):
    if len(elevation) < 2:
        return True
    slopes = [abs((elevation[i+1] - elevation[i]) / 30) for i in range(len(elevation)-1)]
    return max(slopes, default=0) < 0.15

@lru_cache(maxsize=128)
def analyze_terrain_type(coords):
    return True  # Logic moved to batch fetch

def analyze_weather_impact(coords):
    API_KEY = 'f35ddee0eda528827cdb50cd36658ee5'
    BASE_URL = 'http://api.openweathermap.org/data/2.5/weather'
    lat, lon = coords[0][0], coords[0][1]
    try:
        response = requests.get(BASE_URL, params={'lat': lat, 'lon': lon, 'appid': API_KEY, 'units': 'metric'}, timeout=5)
        response.raise_for_status()
        weather_data = response.json()
        weather_main = weather_data['weather'][0]['main'].lower()
        rainfall = weather_data.get('rain', {}).get('1h', 0)
        wind_speed = weather_data['wind']['speed']
        temp = weather_data['main']['temp']
        return not ('rain' in weather_main and rainfall > 5 or 'snow' in weather_main or wind_speed > 10 or temp < 0 or temp > 35)
    except (requests.exceptions.RequestException, KeyError) as e:
        logger.error("Weather impact error: %s", str(e))
        return True

def analyze_obstacles(coords, osm_data):
    route_line = LineString(coords)
    for element in osm_data["elements"]:
        if element.get("tags", {}).get("natural") == "water":
            water_coords = [(node["lat"], node["lon"]) for node in element["geometry"]]
            if route_line.intersects(Polygon(water_coords)):
                return False
    return True

def analyze_emergency_risk(coords, emergency_locations):
    route_line = LineString(coords)
    for loc in emergency_locations:
        if route_line.distance(Point(loc["lon"], loc["lat"])) < 0.01:
            return False
    return True

def analyze_wildlife_threats(coords):
    OVERPASS_URL = "http://overpass-api.de/api/interpreter"
    route_line = LineString(coords)
    south, west, north, east = [x + (0.01 if i > 1 else -0.01) for i, x in enumerate(route_line.bounds)]
    query = f"[out:json];(way['natural'='wood']({south},{west},{north},{east});way['natural'='forest']({south},{west},{north},{east});way['natural'='wetland']({south},{west},{north},{east});way['natural'='scrub']({south},{west},{north},{east}););out geom;"
    try:
        response = requests.post(OVERPASS_URL, data=query, timeout=5)
        response.raise_for_status()
        osm_data = response.json()
        for element in osm_data.get("elements", []):
            if element["type"] == "way" and "geometry" in element:
                feature_coords = [(node["lat"], node["lon"]) for node in element["geometry"]]
                if route_line.distance(Point(feature_coords[0])) < 0.01:
                    logger.info("Wildlife threat: %s at %s", element["tags"]["natural"], feature_coords[0])
                    return False
        return True
    except (requests.exceptions.RequestException, KeyError) as e:
        logger.error("Wildlife threats error: %s", str(e))
        return True

def verify_token(token):
    return True
if __name__ == '__main__':
    app.run(debug=True)