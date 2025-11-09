import geemap
import folium
import ee
from folium.plugins import Draw, Search

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-prathmeshkl2003')

# Define the bounding box for Nashik
nashik_bounds = ee.Geometry.Polygon(
    [[[73.9398, 20.4255],  
      [73.9398, 20.5255],  
      [74.0398, 20.5255],   
      [74.0398, 20.4255]]]) 

# Get the coordinates of the bounds
points = nashik_bounds.coordinates().getInfo()

# Create a folium map centered on Nashik
m = folium.Map(location=[20.4755, 73.9898], zoom_start=8)

# Create a FeatureGroup to hold the points for searching
searchable_points = folium.FeatureGroup(name='Searchable Points').add_to(m)

# Loop through each coordinate to fetch Sentinel-2 Harmonized images and add to the map
for coord in points[0]:
    lon, lat = coord
    point = ee.Geometry.Point([lon, lat])
    
    # Fetch Sentinel-2 Harmonized image
    image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
               .filterBounds(point) \
               .filterDate('2024-01-01', '2024-09-30') \
               .sort('CLOUDY_PIXEL_PERCENTAGE') \
               .first()
    
    # Visualization parameters for true color
    vis_params = {
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2']
    }

    # Use geemap's ee_tile_layer to get tile layer and add it to the map
    tile_layer = geemap.ee_tile_layer(image, vis_params, f'Sentinel-2 ({lat}, {lon})')
    
    # Add the tile layer to the folium map using folium.TileLayer
    folium.TileLayer(
        tiles=tile_layer.url_format,
        attr='Google Earth Engine',
        overlay=True,
        name=f'Sentinel-2 True Color ({lat}, {lon})'
    ).add_to(m)

    # Add marker points to the searchable feature group
    folium.Marker(location=[lat, lon], popup=f"Location: ({lat}, {lon})").add_to(searchable_points)

# Add layer control to the map
folium.LayerControl().add_to(m)

# Add search control (search for marker points in the FeatureGroup)
search = Search(layer=searchable_points, geom_type='Point', placeholder="Search for locations").add_to(m)

# Add drawing tools (toolbar for drawing shapes)
Draw(export=True).add_to(m)

# Save the map as an interactive HTML file with full features
m.save('nashik_map_with_search_and_controls.html')

# Optionally display the map in Jupyter Notebook (if applicable)
# m
