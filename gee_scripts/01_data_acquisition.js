// ======================================================
// Google Earth Engine Script 1: Data Acquisition
// Study Area: Indore District, Madhya Pradesh
// ======================================================

// Step 1: Define Area of Interest (AOI)
// Option 1: Draw polygon manually using geometry tools
// Option 2: Import from shapefile (commented below)

// For Indore District - Manual coordinates
var indore = ee.Geometry.Polygon([
  [[75.65, 22.88], [75.65, 22.52], [76.05, 22.52], [76.05, 22.88]]
]);

// Option 2: Import from uploaded shapefile (uncomment if you have shapefile)
// var indore = ee.FeatureCollection("users/YOUR_USERNAME/indore_boundary");
// var indore = indore.geometry();

// Center map on Indore
Map.centerObject(indore, 10);
Map.addLayer(indore, {color: 'red'}, 'Indore District AOI');

// Step 2: Load Sentinel-2 Surface Reflectance Imagery
var sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(indore)
  .filterDate('2023-01-01', '2023-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .select(['B2', 'B3', 'B4', 'B8']);  // Blue, Green, Red, NIR

print('Total images found:', sentinel2.size());

// Step 3: Create median composite (reduces cloud effects)
var composite = sentinel2.median().clip(indore);

// Step 4: Visualize RGB Composite
var rgbVis = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 3000,
  gamma: 1.4
};

Map.addLayer(composite, rgbVis, 'Sentinel-2 RGB - Indore');

// Step 5: Visualize False Color Composite (NIR, Red, Green)
var falseColorVis = {
  bands: ['B8', 'B4', 'B3'],
  min: 0,
  max: 3500,
  gamma: 1.2
};

Map.addLayer(composite, falseColorVis, 'False Color Composite', false);

// Step 6: Export imagery to Google Drive
Export.image.toDrive({
  image: composite,
  description: 'Indore_Sentinel2_Composite',
  folder: 'GEE_Exports',
  region: indore,
  scale: 10,  // 10m resolution
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

print('Export task created. Check Tasks tab to run the export.');

// Step 7: Display image properties
print('Image bands:', composite.bandNames());
print('Image properties:', composite.getInfo());
