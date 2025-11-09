import pydeck as pdk

# Set the location and initial view
latitude, longitude = 19.9975, 73.7898  # Example: Nashik, India

# View settings with high zoom and 3D pitch
view_state = pdk.ViewState(
    latitude=latitude,
    longitude=longitude,
    zoom=17,      # High zoom
    pitch=60,     # Tilt for 3D effect
    bearing=0
)

# Use OpenStreetMap tiles for the base map
base_map = pdk.Layer(
    "TileLayer",  # Use TileLayer for OpenStreetMap
    data=None,
    get_tile_data="https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
    pickable=True,
)

# Use TerrainLayer for elevation data
terrain_layer = pdk.Layer(
    "TerrainLayer",
    elevation_decoder={
        'rScaler': 256, 'gScaler': 1, 'bScaler': 1/256, 'offset': -32768
    },
    texture=None,  # No texture for now
    elevation_data="https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
    wireframe=False
)

# Render the deck
deck = pdk.Deck(
    layers=[base_map, terrain_layer],  # Combine base map and terrain layers
    initial_view_state=view_state,
    map_style="light",  # Use a simple light style (no Mapbox required)
)

# Save the 3D map as HTML
deck.to_html('3d_map_with_osm.html', notebook_display=False)
print("3D map saved as '3d_map_with_osm.html'")