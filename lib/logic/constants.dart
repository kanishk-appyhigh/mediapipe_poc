import 'models.dart';

OptionsFace options = OptionsFace(
  numClasses: 1,
  numBoxes: 896,
  numCoords: 16,
  keypointCoordOffset: 4,
  ignoreClasses: [],
  scoreClippingThresh: 100.0,
  minScoreThresh: 0.75,
  numKeypoints: 6,
  numValuesPerKeypoint: 2,
  reverseOutputOrder: false,
  boxCoordOffset: 0,
  xScale: 128,
  yScale: 128,
  hScale: 128,
  wScale: 128,
);

AnchorOption anchorOptions = AnchorOption(
  inputSizeHeight: 128,
  inputSizeWidth: 128,
  minScale: 0.1484375,
  maxScale: 0.75,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  numLayers: 4,
  featureMapHeight: [],
  featureMapWidth: [],
  strides: [8, 16, 16, 16],
  aspectRatios: [1.0],
  reduceBoxesInLowestLayer: false,
  interpolatedScaleAspectRatio: 1.0,
  fixedAnchorSize: true,
);

const int imageWidth = 128;
const int imageHeight = 128;
const double minSuppressionThreshold = 0.3;

const double xSCALE = 128.0;
const double ySCALE = 128.0;
const double hSCALE = 128.0;
const double wSCALE = 128.0;
