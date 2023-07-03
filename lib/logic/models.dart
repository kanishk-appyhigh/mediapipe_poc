import 'dart:math';

class OptionsFace {
  final int numClasses;
  final int numBoxes;
  final int numCoords;
  final int keypointCoordOffset;
  final List<int> ignoreClasses;
  final double scoreClippingThresh;
  final double minScoreThresh;
  final int numKeypoints;
  final int numValuesPerKeypoint;
  final int boxCoordOffset;
  final double xScale;
  final double yScale;
  final double wScale;
  final double hScale;
  final bool applyExponentialOnBoxSize;
  final bool reverseOutputOrder;
  final bool sigmoidScore;
  final bool flipVertically;

  OptionsFace({
    required this.numClasses,
    required this.numBoxes,
    required this.numCoords,
    required this.keypointCoordOffset,
    required this.ignoreClasses,
    required this.scoreClippingThresh,
    required this.minScoreThresh,
    this.numKeypoints = 0,
    this.numValuesPerKeypoint = 2,
    this.boxCoordOffset = 0,
    this.xScale = 0.0,
    this.yScale = 0.0,
    this.wScale = 0.0,
    this.hScale = 0.0,
    this.applyExponentialOnBoxSize = false,
    this.reverseOutputOrder = true,
    this.sigmoidScore = true,
    this.flipVertically = false,
  });
}

class AnchorOption {
  final int inputSizeWidth;
  final int inputSizeHeight;
  final double minScale;
  final double maxScale;
  final double anchorOffsetX;
  final double anchorOffsetY;
  final int numLayers;
  final List<int> featureMapWidth;
  final List<int> featureMapHeight;
  final List<int> strides;
  final List<double> aspectRatios;
  final bool reduceBoxesInLowestLayer;
  final double interpolatedScaleAspectRatio;
  final bool fixedAnchorSize;

  AnchorOption({
    required this.inputSizeWidth,
    required this.inputSizeHeight,
    required this.minScale,
    required this.maxScale,
    required this.anchorOffsetX,
    required this.anchorOffsetY,
    required this.numLayers,
    required this.featureMapWidth,
    required this.featureMapHeight,
    required this.strides,
    required this.aspectRatios,
    required this.reduceBoxesInLowestLayer,
    required this.interpolatedScaleAspectRatio,
    required this.fixedAnchorSize,
  });

  int get stridesSize {
    return strides.length;
  }

  int get featureMapHeightSize {
    return featureMapHeight.length;
  }

  int get featureMapWidthSize {
    return featureMapWidth.length;
  }
}

class Anchor {
  final double xCenter;
  final double yCenter;
  final double h;
  final double w;

  Anchor(
    this.xCenter,
    this.yCenter,
    this.h,
    this.w,
  );
}

class Detection {
  final double score;
  final int classID;
  final double xMin;
  final double yMin;
  final double width;
  final double height;
  Detection(
    this.score,
    this.classID,
    this.xMin,
    this.yMin,
    this.width,
    this.height,
  );
}

class Rect {
  double left; // xmin
  double right; // xmax
  double top; // ymin
  double bottom; // ymax

  Rect(
    this.left,
    this.right,
    this.top,
    this.bottom,
  );

  bool intersects(Rect other) {
    return left < other.right &&
        other.left < right &&
        top < other.bottom &&
        other.top < bottom;
  }

  double intersection_area(Rect other) {
    double x = max(0.0, min(right, other.right) - max(left, other.left));
    // This is the original, but for some reason it's broke and i'm lazy to
    // debug it. so switch the way you subtract and it works.
    // double y = max(0.0, min(top, other.top) - max(bottom, other.bottom));
    double y = max(0.0, min(bottom, other.bottom) - max(top, other.top));
    return x * y;
  }

  double height() {
    return (top - bottom).abs();
  }

  double width() {
    return (right - left).abs();
  }

  double area() {
    return width() * height();
  }
}

class NewDetection {
  final Rect location;
  final double score;
  NewDetection(
    this.location,
    this.score,
  );
}

class IndexedScore {
  final int index;
  final double score;
  IndexedScore(this.index, this.score);
}
