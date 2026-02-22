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
  .select(['B2', 'B3', 'B4', 'B8'])  // Select only needed bands BEFORE median
  .median()
  .clip(indore);


// Add features
var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');

var compositeSWIR = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(indore)
  .filterDate('2023-01-01', '2023-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .select(['B8', 'B11'])  // Select both bands needed for NDBI
  .median()
  .clip(indore);

var ndbi = compositeSWIR.normalizedDifference(['B11', 'B8'])  // Use string band names, not Image objects
  .rename('NDBI');

var glcm = composite.select('B8').toUint16().glcmTexture({ size: 3 });
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

// ======================================================
// HOW TO CREATE TRAINING SAMPLES IN GEE:
// ======================================================
// 1. In the GEE Code Editor, click the polygon draw tool (top-left of map)
// 2. Click "+ new layer" for each class
// 3. For each layer, click the gear icon ⚙️ and set:
//      Name: buildings   Import as: FeatureCollection   Property: class = 1
//      Name: roads       Import as: FeatureCollection   Property: class = 2
//      Name: water       Import as: FeatureCollection   Property: class = 3
//      Name: vegetation  Import as: FeatureCollection   Property: class = 4
// 4. Draw 10-15 polygons per class on the map
// 5. After drawing, GEE auto-adds them as imports at the top of this script
// 6. Then run the code below (it's already uncommented and ready to use)
// ======================================================

// Merge all training samples
// (GEE auto-imports 'buildings', 'roads', 'water', 'vegetation' from geometry tools)
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

Map.addLayer(classified, { min: 1, max: 4, palette: palette },
  'Classification');

// ======================================================
// STEP 5: EXTRACT INDIVIDUAL CLASS MASKS
// ======================================================

var buildingsMask = classified.eq(1).selfMask();
var roadsMask = classified.eq(2).selfMask();
var waterMask = classified.eq(3).selfMask();
var vegetationMask = classified.eq(4).selfMask();

Map.addLayer(buildingsMask, { palette: ['red'] }, 'Buildings Only', false);
Map.addLayer(roadsMask, { palette: ['blue'] }, 'Roads Only', false);
Map.addLayer(waterMask, { palette: ['cyan'] }, 'Water Only', false);

// ======================================================
// STEP 6: EXPORT CLASSIFICATION RESULTS
// ======================================================

// Export classified image
Export.image.toDrive({
  image: classified,
  description: 'Indore_Classification',
  folder: 'GEE_Exports',
  region: indore,
  scale: 30,  // Increased from 10 to 30m to avoid 'computed value too large' error
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Export individual masks for DL training
Export.image.toDrive({
  image: buildingsMask.toByte(),
  description: 'Indore_Buildings_Mask',
  folder: 'GEE_Exports',
  region: indore,
  scale: 30,  // Increased from 10 to 30m to avoid 'computed value too large' error
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

print('Export tasks created. Check the Tasks tab to run exports.');

print('==================================================');
print('NEXT STEP:');
print('Use geometry tools to draw training samples,');
print('then re-run this script.');
print('==================================================');
