// ======================================================
// Google Earth Engine Script 2: Feature Engineering
// Spectral Indices and Texture Features
// ======================================================

// Load the composite from previous script
// Or re-create it here

var indore = ee.Geometry.Polygon([
    [[75.65, 22.88], [75.65, 22.52], [76.05, 22.52], [76.05, 22.88]]
]);

var composite = ee.ImageCollection("COPERNICUS/S2_SR")
    .filterBounds(indore)
    .filterDate('2023-01-01', '2023-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .median()
    .clip(indore);

Map.centerObject(indore, 10);

// Step 1: Calculate NDVI (Normalized Difference Vegetation Index)
// NDVI = (NIR - Red) / (NIR + Red)
// Values: -1 to 1, higher values = more vegetation
var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');

var ndviVis = {
    min: -0.2,
    max: 0.8,
    palette: ['brown', 'yellow', 'lightgreen', 'darkgreen']
};

Map.addLayer(ndvi, ndviVis, 'NDVI', false);

// Step 2: Calculate NDWI (Normalized Difference Water Index)
// NDWI = (Green - NIR) / (Green + NIR)
// Highlights water bodies
var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');

var ndwiVis = {
    min: -0.5,
    max: 0.5,
    palette: ['white', 'lightblue', 'blue', 'darkblue']
};

Map.addLayer(ndwi, ndwiVis, 'NDWI', false);

// Step 3: Calculate NDBI (Normalized Difference Built-up Index)
// NDBI = (SWIR - NIR) / (SWIR + NIR)
// Note: Using B11 (SWIR1) for built-up areas
var compositeSWIR = ee.ImageCollection("COPERNICUS/S2_SR")
    .filterBounds(indore)
    .filterDate('2023-01-01', '2023-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .select(['B8', 'B11'])
    .median()
    .clip(indore);

var ndbi = compositeSWIR.normalizedDifference(['B11', 'B8']).rename('NDBI');

var ndbiVis = {
    min: -0.5,
    max: 0.5,
    palette: ['green', 'white', 'orange', 'red']
};

Map.addLayer(ndbi, ndbiVis, 'NDBI (Built-up)', false);

// Step 4: Texture Features using GLCM (Gray-Level Co-occurrence Matrix)
// Texture helps distinguish buildings from open areas
var nir = composite.select('B8');

var glcm = nir.glcmTexture({ size: 3 });

// Select useful texture measures
var contrast = glcm.select('B8_contrast').rename('texture_contrast');
var entropy = glcm.select('B8_ent').rename('texture_entropy');

Map.addLayer(contrast, { min: 0, max: 500, palette: ['black', 'white'] },
    'Texture Contrast', false);

// Step 5: Combine all features into enhanced image
var enhancedImage = composite
    .addBands(ndvi)
    .addBands(ndwi)
    .addBands(ndbi)
    .addBands(contrast)
    .addBands(entropy);

print('Enhanced Image Bands:', enhancedImage.bandNames());

// Step 6: Export enhanced multi-band image
Export.image.toDrive({
    image: enhancedImage,
    description: 'Indore_Enhanced_Features',
    folder: 'GEE_Exports',
    region: indore,
    scale: 10,
    crs: 'EPSG:4326',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
});

print('Enhanced image export task created.');
print('Total bands:', enhancedImage.bandNames().size());
