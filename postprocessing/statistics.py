"""
Object Statistics and Analysis
Calculate metrics for detected objects
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from typing import Dict, List
import yaml
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_class_statistics(classification_raster: str,
                               config_path: str = 'config/config.yaml') -> pd.DataFrame:
    """
    Calculate statistics for each class.
    
    Args:
        classification_raster: Path to classification raster
        config_path: Path to configuration file
        
    Returns:
        DataFrame with statistics
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Read raster
    with rasterio.open(classification_raster) as src:
        image = src.read(1)
        pixel_area = abs(src.transform[0] * src.transform[4])  # Area per pixel
    
    # Calculate statistics
    stats = []
    
    for class_info in config['classes']:
        class_id = class_info['id']
        class_name = class_info['name']
        
        # Count pixels
        pixel_count = np.sum(image == class_id)
        area = pixel_count * pixel_area
        percentage = (pixel_count / image.size) * 100
        
        stats.append({
            'Class ID': class_id,
            'Class Name': class_name,
            'Pixel Count': pixel_count,
            'Area (sq m)': area,
            'Area (sq km)': area / 1e6,
            'Percentage': percentage
        })
    
    df = pd.DataFrame(stats)
    
    return df


def calculate_vector_statistics(vector_dir: str,
                                config_path: str = 'config/config.yaml') -> pd.DataFrame:
    """
    Calculate statistics from vector files.
    
    Args:
        vector_dir: Directory with vector files
        config_path: Path to configuration file
        
    Returns:
        DataFrame with statistics
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vector_dir = Path(vector_dir)
    
    stats = []
    
    for class_info in config['classes']:
        class_id = class_info['id']
        class_name = class_info['name']
        
        if class_id == 0:  # Skip background
            continue
        
        vector_file = vector_dir / f'{class_name}.shp'
        
        if not vector_file.exists():
            continue
        
        # Read vector file
        gdf = gpd.read_file(vector_file)
        
        if len(gdf) == 0:
            continue
        
        # Calculate statistics
        total_area = gdf.geometry.area.sum()
        mean_area = gdf.geometry.area.mean()
        median_area = gdf.geometry.area.median()
        std_area = gdf.geometry.area.std()
        
        stats.append({
            'Class Name': class_name,
            'Object Count': len(gdf),
            'Total Area (sq m)': total_area,
            'Total Area (sq km)': total_area / 1e6,
            'Mean Area (sq m)': mean_area,
            'Median Area (sq m)': median_area,
            'Std Area (sq m)': std_area,
            'Min Area (sq m)': gdf.geometry.area.min(),
            'Max Area (sq m)': gdf.geometry.area.max()
        })
    
    df = pd.DataFrame(stats)
    
    return df


def calculate_density(vector_file: str, study_area_sqkm: float) -> float:
    """
    Calculate object density.
    
    Args:
        vector_file: Path to vector file
        study_area_sqkm: Study area in square kilometers
        
    Returns:
        Density (objects per sq km)
    """
    gdf = gpd.read_file(vector_file)
    count = len(gdf)
    density = count / study_area_sqkm
    
    return density


def plot_class_distribution(stats_df: pd.DataFrame, output_path: str):
    """
    Plot class distribution.
    
    Args:
        stats_df: Statistics DataFrame
        output_path: Output path for plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Pixel count
    ax = axes[0]
    sns.barplot(data=stats_df, x='Class Name', y='Pixel Count', ax=ax)
    ax.set_title('Pixel Count by Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Pixel Count')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: Area percentage
    ax = axes[1]
    colors = ['#%02x%02x%02x' % tuple(c['color']) for c in stats_df.to_dict('records')]
    ax.pie(stats_df['Percentage'], labels=stats_df['Class Name'], autopct='%1.1f%%',
           startangle=90)
    ax.set_title('Area Distribution by Class')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")


def generate_report(classification_raster: str, vector_dir: str,
                   output_dir: str, config_path: str = 'config/config.yaml'):
    """
    Generate comprehensive statistics report.
    
    Args:
        classification_raster: Path to classification raster
        vector_dir: Directory with vector files
        output_dir: Output directory
        config_path: Path to configuration file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Statistics Report")
    print("=" * 60)
    
    # Raster statistics
    print("\nCalculating raster statistics...")
    raster_stats = calculate_class_statistics(classification_raster, config_path)
    
    print("\nRaster Statistics:")
    print(raster_stats.to_string(index=False))
    
    # Save to CSV
    raster_stats.to_csv(output_dir / 'raster_statistics.csv', index=False)
    
    # Vector statistics
    print("\nCalculating vector statistics...")
    vector_stats = calculate_vector_statistics(vector_dir, config_path)
    
    if len(vector_stats) > 0:
        print("\nVector Statistics:")
        print(vector_stats.to_string(index=False))
        
        # Save to CSV
        vector_stats.to_csv(output_dir / 'vector_statistics.csv', index=False)
    
    # Plot
    print("\nGenerating plots...")
    plot_class_distribution(raster_stats, output_dir / 'class_distribution.png')
    
    # Generate summary
    summary_file = output_dir / 'summary_report.txt'
    
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("OBJECT DETECTION STATISTICS REPORT\n")
        f.write("Deep Learning-Based Analysis - Indore District\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("RASTER STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(raster_stats.to_string(index=False))
        f.write("\n\n")
        
        if len(vector_stats) > 0:
            f.write("VECTOR STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(vector_stats.to_string(index=False))
            f.write("\n\n")
    
    print(f"\nReport saved to: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate object statistics')
    parser.add_argument('--raster', required=True, help='Classification raster')
    parser.add_argument('--vectors', required=True, help='Vector directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    generate_report(args.raster, args.vectors, args.output, args.config)
