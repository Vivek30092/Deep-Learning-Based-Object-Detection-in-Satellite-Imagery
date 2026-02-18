// ======================================================
// Google Earth Engine Script 3: AI-Assisted Classification
// Random Forest Classifier for Training Data Generation
// ======================================================

// Load enhanced image from previous script
var indore = ee.Geometry.Polygon([
    [[75.65, 22.88], [75.65, 22.52], [76.05, 22.52], [76.05, 22.88]]
]);

var composite = ee.ImageCollection("COPERNICUS/S2_SR")
    .filterBounds(indore)
    .filterDate('2023-01-01', '2023-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .median()
    .clip(indore);

// Add features
var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');

var compositeSWIR = ee.ImageCollection("COPERNICUS/S2_SR")
    .filterBounds(indore)
    .filterDate('2023-01-01', '2023-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .select(['B11'])
    .median()
    .clip(indore);

var ndbi = compositeSWIR.normalizedDifference(['B11', composite.select('B8')])
    .rename('NDBI');

var glcm = composite.select('B8').glcmTexture({ size: 3 });
var contrast = glcm.select('B8_contrast').rename('texture_contrast');

var finalImage = composite
    .addBands(ndvi)
    .addBands(ndwi)
    .addBands(ndbi)
    .addBands(contrast);

Map.centerObject(indore, 11);
Map.addLayer(composite, { bands: ['B4', 'B3', 'B2'], min: 0, max: 3000 }, 'RGB');

// ======================================================
// STEP 1: CREATE TRAINING SAMPLES
// ======================================================
// INSTRUCTIONS:
// 1. Use the geometry tools to digitize training samples
// 2. Create separate geometry imports for each class:
//    - buildings (red polygons)
//    - roads (blue polylines/polygons)
//    - water (cyan polygons)
//    - vegetation (green polygons)
//    - bare_soil (brown polygons)
//
// 3. After digitizing, uncomment the code below

/*
// Example training data structure
// Replace with your actual digitized geometries

var buildings = ee.FeatureCollection([
  ee.Feature(geometry1, {class: 1}),
  ee.Feature(geometry2, {class: 1})
]);

var roads = ee.FeatureCollection([
  ee.Feature(geometry3, {class: 2})
]);

var water = ee.FeatureCollection([
  ee.Feature(geometry4, {class: 3})
]);

var vegetation = ee.FeatureCollection([
  ee.Feature(geometry5, {class: 4})
]);

// Merge all training samples
var trainingSamples = buildings
  .merge(roads)
  .merge(water)
  .merge(vegetation);

print('Training samples count:', trainingSamples.size());

// ======================================================
// STEP 2: SAMPLE REGIONS
// ======================================================

var bands = finalImage.bandNames();

var training = finalImage.sampleRegions({
  collection: trainingSamples,
  properties: ['class'],
  scale: 10
});

print('Training points:', training.size());

// ======================================================
// STEP 3: TRAIN RANDOM FOREST CLASSIFIER
// ======================================================

var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  variablesPerSplit: null,
  minLeafPopulation: 1,
  bagFraction: 0.5,
  seed: 42
}).train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// ======================================================
// STEP 4: CLASSIFY IMAGE
// ======================================================

var classified = finalImage.classify(classifier);

// Define color palette
var palette = [
  'black',      // 0 - background (not used)
  'red',        // 1 - buildings
  'blue',       // 2 - roads
  'cyan',       // 3 - water
  'green'       // 4 - vegetation
];

Map.addLayer(classified, {min: 1, max: 4, palette: palette}, 
             'Classification');

// ======================================================
// STEP 5: EXTRACT INDIVIDUAL CLASS MASKS
// ======================================================

var buildingsMask = classified.eq(1).selfMask();
var roadsMask = classified.eq(2).selfMask();
var waterMask = classified.eq(3).selfMask();
var vegetationMask = classified.eq(4).selfMask();

Map.addLayer(buildingsMask, {palette: ['red']}, 'Buildings Only', false);
Map.addLayer(roadsMask, {palette: ['blue']}, 'Roads Only', false);
Map.addLayer(waterMask, {palette: ['cyan']}, 'Water Only', false);

// ======================================================
// STEP 6: EXPORT CLASSIFICATION RESULTS
// ======================================================

// Export classified image
Export.image.toDrive({
  image: classified,
  description: 'Indore_Classification',
  folder: 'GEE_Exports',
  region: indore,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Export individual masks for DL training
Export.image.toDrive({
  image: buildingsMask.toByte(),
  description: 'Indore_Buildings_Mask',
  folder: 'GEE_Exports',
  region: indore,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

print('Export tasks created. Digitize training samples and uncomment code to run.');
*/

print('='.repeat(50));
print('INSTRUCTIONS:');
print('1. Use geometry tools to digitize training samples');
print('2. Create geometries for: buildings, roads, water, vegetation');
print('3. Uncomment the code section above');
print('4. Run the script again');
print('='.repeat(50));
