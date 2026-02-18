"""
Raster to Vector Conversion
Convert segmentation masks to GIS vector formats
"""

import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pathlib import Path
from typing import List, Optional
import yaml


def polygonize_raster(raster_path: str, output_path: str,
                     class_value: Optional[int] = None,
                     simplify_tolerance: float = 2.0,
                     min_area: float = 100.0) -> gpd.GeoDataFrame:
    """
    Convert raster to vector polygons.
    
    Args:
        raster_path: Path to input raster
        output_path: Path to output vector file
        class_value: Specific class value to extract (None = all non-zero)
        simplify_tolerance: Douglas-Peucker simplification tolerance
        min_area: Minimum polygon area in pixels
        
    Returns:
        GeoDataFrame with polygons
    """
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs
        
        # Create mask
        if class_value is not None:
            mask = image == class_value
        else:
            mask = image > 0
        
        # Extract shapes
        shapes = features.shapes(
            mask.astype(np.uint8),
            mask=mask,
            transform=transform
        )
        
        # Convert to geometries
        geometries = []
        values = []
        
        for geom, value in shapes:
            if value == 1:  # Only keep positive features
                polygon = shape(geom)
                
                # Filter by area
                if polygon.area >= min_area:
                    # Simplify
                    if simplify_tolerance > 0:
                        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)
                    
                    geometries.append(polygon)
                    values.append(class_value if class_value else 1)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'class': values, 'area': [g.area for g in geometries]},
            geometry=geometries,
            crs=crs
        )
        
        # Save
        output_path = Path(output_path)
        
        if output_path.suffix == '.shp':
            gdf.to_file(output_path, driver='ESRI Shapefile')
        elif output_path.suffix == '.geojson':
            gdf.to_file(output_path, driver='GeoJSON')
        elif output_path.suffix == '.gpkg':
            gdf.to_file(output_path, driver='GPKG')
        else:
            gdf.to_file(output_path)
        
        print(f"Saved {len(gdf)} polygons to: {output_path}")
        
        return gdf


def convert_all_classes(classification_raster: str, output_dir: str,
                       config_path: str = 'config/config.yaml'):
    """
    Convert all classes from classification raster to vector.
    
    Args:
        classification_raster: Path to classification GeoTIFF
        output_dir: Output directory for vector files
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    postproc_config = config.get('postprocessing', {})
    
    print("=" * 60)
    print("Converting Raster to Vector")
    print("=" * 60)
    
    # Process each class
    for class_info in config['classes']:
        class_id = class_info['id']
        class_name = class_info['name']
        
        if class_id == 0:  # Skip background
            continue
        
        print(f"\nProcessing class: {class_name} (ID: {class_id})")
        
        output_path = output_dir / f'{class_name}.shp'
        
        gdf = polygonize_raster(
            classification_raster,
            str(output_path),
            class_value=class_id,
            simplify_tolerance=postproc_config.get('simplify_tolerance', 2.0),
            min_area=postproc_config.get('min_area', 100.0)
        )
        
        if len(gdf) > 0:
            print(f"  Total area: {gdf['area'].sum():.2f} sq units")
            print(f"  Mean area: {gdf['area'].mean():.2f} sq units")
            print(f"  Number of features: {len(gdf)}")


def merge_adjacent_polygons(gdf: gpd.GeoDataFrame, buffer_distance: float = 1.0) -> gpd.GeoDataFrame:
    """
    Merge adjacent polygons.
    
    Args:
        gdf: Input GeoDataFrame
        buffer_distance: Buffer distance for merging
        
    Returns:
        GeoDataFrame with merged polygons
    """
    # Buffer, union, and reverse buffer
    buffered = gdf.geometry.buffer(buffer_distance)
    merged = unary_union(buffered)
    final = merged.buffer(-buffer_distance)
    
    # Create new GeoDataFrame
    if final.geom_type == 'Polygon':
        geometries = [final]
    else:
        geometries = list(final.geoms)
    
    merged_gdf = gpd.GeoDataFrame(
        {'class': [gdf['class'].iloc[0]] * len(geometries)},
        geometry=geometries,
        crs=gdf.crs
    )
    
    return merged_gdf


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert raster to vector')
    parser.add_argument('--input', required=True, help='Input classification raster')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    convert_all_classes(args.input, args.output, args.config)
