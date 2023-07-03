import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:logger/logger.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

import 'package:image/image.dart' as img;

import 'constants.dart';
import 'models.dart';

class FaceDetectionHelper {
  final Logger _logger = Logger();

  static int detectFace(
    final RootIsolateToken rootIsolateToken,
    Uint8List thumbnail,
    tfl.Interpreter interpreter,
  ) {
    BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);
    print(interpreter == null);

    Float32List float32List = Float32List.fromList(
      thumbnail.map((value) => (value.toDouble() - 127.5) / 127.5).toList(),
    );

    Map<int, Object> output = {
      0: List.filled(16 * 896, 0.0).reshape([1, 896, 16]),
      1: List.filled(896, 0.0).reshape([1, 896, 1]),
    };

    interpreter.runForMultipleInputs([
      float32List.reshape([1, 128, 128, 3])
    ], output);

    List regression = output[0] as List;
    List classificators = output[1] as List;

    List<Anchor> anchors = _getAnchors(anchorOptions);

    int detectionsLength = _processRaw(
      boxes: regression.reshape([896 * 16]),
      scores: classificators.reshape([896]),
      anchorList: anchors,
    );

    // _logger.d("Faces detected: $detectionsLength");

    return detectionsLength;
  }

  static int _processRaw({
    required List scores,
    required List boxes,
    required List<Anchor> anchorList,
  }) {
    List<NewDetection> detections = [];
    List<IndexedScore> indexedScoreList = [];

    for (var i = 0; i < 896; i++) {
      double score = scores[i];
      score = max(score, -100.0);
      score = min(score, 100.0);
      score = 1.0 / (1.0 + exp(-score));

      if (score <= 0.75) continue;

      double xCenter = boxes[i * 16];
      double yCenter = boxes[i * 16 + 1];
      double width = boxes[i * 16 + 2];
      double height = boxes[i * 16 + 3];

      xCenter = xCenter / xSCALE * anchorList[i].w + anchorList[i].xCenter;
      yCenter = yCenter / ySCALE * anchorList[i].h + anchorList[i].yCenter;
      height = height / hSCALE * anchorList[i].h;
      width = width / wSCALE * anchorList[i].w;

      double ymin = min(yCenter - height / 2.0, 1.0);
      double xmin = min(xCenter - width / 2.0, 1.0);
      double ymax = min(yCenter + height / 2.0, 1.0);
      double xmax = min(xCenter + width / 2.0, 1.0);

      Rect location = Rect(xmin, xmax, ymin, ymax);
      NewDetection det = NewDetection(location, score);
      detections.add(det);
    }

    if (detections.isEmpty) {
      return 0;
    }

    for (int index = 0; index < detections.length; ++index) {
      IndexedScore indexedScore = IndexedScore(index, detections[index].score);
      indexedScoreList.add(indexedScore);
    }

    indexedScoreList.sort((a, b) => -a.score.compareTo(b.score));

    List<Rect> finalDets =
        _weightedNonMaxSuppression(detections, indexedScoreList);

    return finalDets.length;
  }

  static List<Rect> _weightedNonMaxSuppression(
      List<NewDetection> detections, List<IndexedScore> indexedScores) {
    List<IndexedScore> remained = [];
    List<IndexedScore> candidates = [];
    List<Rect> outputLocs = [];

    while (indexedScores.isNotEmpty) {
      NewDetection det = detections[indexedScores[0].index];

      remained.clear();
      candidates.clear();

      Rect loc = det.location;
      for (var idxScore in indexedScores) {
        Rect restLoc = detections[idxScore.index].location;
        double similarity = _overlapSimilarity(restLoc, loc);

        if (similarity > minSuppressionThreshold) {
          candidates.add(idxScore);
        } else {
          remained.add(idxScore);
        }
      }

      if (candidates.isNotEmpty) {
        double wxmin = 0.0;
        double wymin = 0.0;
        double wxmax = 0.0;
        double wymax = 0.0;
        double totalScore = 0.0;

        for (var candidate in candidates) {
          totalScore += candidate.score;
          Rect bbox = detections[candidate.index].location;
          wxmin += bbox.left * candidate.score;
          wymin += bbox.top * candidate.score;
          wxmax += bbox.right * candidate.score;
          wymax += bbox.bottom * candidate.score;
        }

        loc.left = wxmin / totalScore * imageWidth;
        loc.top = wymin / totalScore * imageHeight;
        loc.right = wxmax / totalScore * imageWidth;
        loc.bottom = wymax / totalScore * imageHeight;
      }

      indexedScores.clear();
      indexedScores.addAll(remained);
      outputLocs.add(loc);
    }
    return outputLocs;
  }

  static double _overlapSimilarity(Rect rect1, Rect rect2) {
    bool intersects = rect1.intersects(rect2);
    if (!intersects) return 0.0;

    double intersectionArea = rect1.intersection_area(rect2);
    double normalization = rect1.height() * rect1.width() +
        rect2.height() * rect2.width() -
        intersectionArea;

    return normalization > 0.0 ? intersectionArea / normalization : 0.0;
  }

  static List<Anchor> _getAnchors(AnchorOption options) {
    List<Anchor> anchors = [];
    if (options.stridesSize != options.numLayers) {
      return [];
    }
    int layerID = 0;

    while (layerID < options.stridesSize) {
      List<double> anchorHeight = [];
      List<double> anchorWidth = [];
      List<double> aspectRatios = [];
      List<double> scales = [];

      int lastSameStrideLayer = layerID;
      while (lastSameStrideLayer < options.stridesSize &&
          options.strides[lastSameStrideLayer] == options.strides[layerID]) {
        double scale = options.minScale +
            (options.maxScale - options.minScale) *
                1.0 *
                lastSameStrideLayer /
                (options.stridesSize - 1.0);
        if (lastSameStrideLayer == 0 && options.reduceBoxesInLowestLayer) {
          aspectRatios.add(1.0);
          aspectRatios.add(2.0);
          aspectRatios.add(0.5);
          scales.add(0.1);
          scales.add(scale);
          scales.add(scale);
        } else {
          for (int i = 0; i < options.aspectRatios.length; i++) {
            aspectRatios.add(options.aspectRatios[i]);
            scales.add(scale);
          }

          if (options.interpolatedScaleAspectRatio > 0.0) {
            double scaleNext = 0.0;
            if (lastSameStrideLayer == options.stridesSize - 1) {
              scaleNext = 1.0;
            } else {
              scaleNext = options.minScale +
                  (options.maxScale - options.minScale) *
                      1.0 *
                      (lastSameStrideLayer + 1) /
                      (options.stridesSize - 1.0);
            }
            scales.add(sqrt(scale * scaleNext));
            aspectRatios.add(options.interpolatedScaleAspectRatio);
          }
        }
        lastSameStrideLayer++;
      }
      for (int i = 0; i < aspectRatios.length; i++) {
        double ratioSQRT = sqrt(aspectRatios[i]);
        anchorHeight.add(scales[i] / ratioSQRT);
        anchorWidth.add(scales[i] * ratioSQRT);
      }
      int featureMapHeight = 0;
      int featureMapWidth = 0;
      if (options.featureMapHeightSize > 0) {
        featureMapHeight = options.featureMapHeight[layerID];
        featureMapWidth = options.featureMapWidth[layerID];
      } else {
        int stride = options.strides[layerID];
        featureMapHeight = (1.0 * options.inputSizeHeight / stride).ceil();
        featureMapWidth = (1.0 * options.inputSizeWidth / stride).ceil();
      }

      for (int y = 0; y < featureMapHeight; y++) {
        for (int x = 0; x < featureMapWidth; x++) {
          for (int anchorID = 0; anchorID < anchorHeight.length; anchorID++) {
            double xCenter =
                (x + options.anchorOffsetX) * 1.0 / featureMapWidth;
            double yCenter =
                (y + options.anchorOffsetY) * 1.0 / featureMapHeight;
            double w = 0;
            double h = 0;
            if (options.fixedAnchorSize) {
              w = 1.0;
              h = 1.0;
            } else {
              w = anchorWidth[anchorID];
              h = anchorHeight[anchorID];
            }
            anchors.add(Anchor(xCenter, yCenter, h, w));
          }
        }
      }
      layerID = lastSameStrideLayer;
    }
    return anchors;
  }
}

class ImagePickerHelper {
  Future<XFile?> pickImageFromGallery() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    return image;
  }

  Future<Uint8List?> resizeImage(XFile pickedImage) async {
    // Read image from the file.
    Uint8List imageBytes = await pickedImage.readAsBytes();
    img.Image? image = img.decodeImage(imageBytes);

    // Resize the image to a 128x128 thumbnail (maintaining the aspect ratio).
    img.Image thumbnail = img.copyResize(image!, width: 128, height: 128);

    return thumbnail.getBytes();
  }
}


// import 'dart:math';
// import 'dart:typed_data';

// import 'package:flutter/services.dart';
// import 'package:image_picker/image_picker.dart';
// import 'package:logger/logger.dart';
// import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

// import 'package:image/image.dart' as img;

// import 'defaults.dart';
// import 'models.dart';

// class FaceDetectionHelper {
//   static tfl.Interpreter? _interpreter;
//   final _logger = Logger();

//   Future<void> initialiseModel() async {
//     try {
//       final tfl.Interpreter interpreter =
//           await tfl.Interpreter.fromAsset('assets/models/blazeface.tflite');

//       _interpreter = interpreter;
//     } on PlatformException {
//       print('Failed to load model.');
//     } catch (e) {
//       print(e);
//     }
//   }

//   Future<int> detectFace(Uint8List thumbnail) async {
//     try {
//       isolatedGrid = await Isolate.run<int?>(
//           () => _detectFace(rootIsolateToken));
//     } catch (e, s) {
//       AppToast.showToast("Oops! Something went wrong, please try again later.");
//       _logger.e("Error occurred while spawing an Isolate");
//       _logger.e(e);
//       _logger.e(s);
//     }
//   }

//   Future<int> _detectFace(Uint8List thumbnail) async {
//     if (_interpreter == null) return 0;

//     Float32List float32List = Float32List.fromList(
//       thumbnail.map((value) => (value.toDouble() - 127.5) / 127.5).toList(),
//     );

//     Map<int, Object> output = {
//       0: List.filled(16 * 896, 0.0).reshape([1, 896, 16]),
//       1: List.filled(896, 0.0).reshape([1, 896, 1]),
//     };

//     _interpreter!.runForMultipleInputs([
//       float32List.reshape([1, 128, 128, 3])
//     ], output);

//     List regression = output[0] as List;
//     List classificators = output[1] as List;

//     List<Anchor> anchors = _getAnchors(anchorOptions);

//     int detectionsLength = _processRaw(
//       boxes: regression.reshape([896 * 16]),
//       scores: classificators.reshape([896]),
//       anchorList: anchors,
//     );

//     return detectionsLength;
//   }

//   int _processRaw({
//     required List scores,
//     required List boxes,
//     required List<Anchor> anchorList,
//   }) {
//     const double X_SCALE = 128.0;
//     const double Y_SCALE = 128.0;
//     const double H_SCALE = 128.0;
//     const double W_SCALE = 128.0;

//     List<NewDetection> detections = [];
//     List<IndexedScore> indexed_score_list = [];

//     for (var i = 0; i < 896; i++) {
//       double score = scores[i];
//       score = max(score, -100.0);
//       score = min(score, 100.0);
//       score = 1.0 / (1.0 + exp(-score));

//       if (score <= 0.75) continue;

//       double x_center = boxes[i * 16];
//       double y_center = boxes[i * 16 + 1];
//       double width = boxes[i * 16 + 2];
//       double height = boxes[i * 16 + 3];

//       x_center = x_center / X_SCALE * anchorList[i].w + anchorList[i].xCenter;
//       y_center = y_center / Y_SCALE * anchorList[i].h + anchorList[i].yCenter;
//       height = height / H_SCALE * anchorList[i].h;
//       width = width / W_SCALE * anchorList[i].w;

//       double ymin = min(y_center - height / 2.0, 1.0);
//       double xmin = min(x_center - width / 2.0, 1.0);
//       double ymax = min(y_center + height / 2.0, 1.0);
//       double xmax = min(x_center + width / 2.0, 1.0);

//       Rect location = Rect(xmin, xmax, ymin, ymax);
//       NewDetection det = NewDetection(location, score);
//       detections.add(det);
//     }

//     if (detections.isEmpty) {
//       return 0;
//     }

//     for (int index = 0; index < detections.length; ++index) {
//       IndexedScore indexed_score = IndexedScore(index, detections[index].score);
//       indexed_score_list.add(indexed_score);
//     }

//     indexed_score_list.sort((a, b) => -a.score.compareTo(b.score));

//     List<Rect> final_dets =
//         _weightedNonMaxSuppression(detections, indexed_score_list);

//     return final_dets.length;
//   }

//   List<Rect> _weightedNonMaxSuppression(
//       List<NewDetection> detections, List<IndexedScore> indexed_scores) {
//     const int IMAGE_WIDTH = 128;
//     const int IMAGE_HEIGHT = 128;
//     const double MIN_SUPPRESSION_THRESHOLD = 0.3;

//     List<IndexedScore> remained = [];
//     List<IndexedScore> candidates = [];
//     List<Rect> output_locs = [];

//     while (indexed_scores.isNotEmpty) {
//       NewDetection det = detections[indexed_scores[0].index];

//       remained.clear();
//       candidates.clear();

//       Rect loc = det.location;
//       // for (int i = 1; i < indexed_scores.length; i++) {
//       // IndexedScore idx_score = indexed_scores[i];
//       for (var idx_score in indexed_scores) {
//         Rect rest_loc = detections[idx_score.index].location;
//         double similarity = _overlapSimilarity(rest_loc, loc);

//         if (similarity > MIN_SUPPRESSION_THRESHOLD) {
//           candidates.add(idx_score);
//         } else {
//           remained.add(idx_score);
//         }
//       }

//       if (candidates.isNotEmpty) {
//         double w_xmin = 0.0;
//         double w_ymin = 0.0;
//         double w_xmax = 0.0;
//         double w_ymax = 0.0;
//         double total_score = 0.0;

//         for (var candidate in candidates) {
//           total_score += candidate.score;
//           Rect bbox = detections[candidate.index].location;
//           w_xmin += bbox.left * candidate.score;
//           w_ymin += bbox.top * candidate.score;
//           w_xmax += bbox.right * candidate.score;
//           w_ymax += bbox.bottom * candidate.score;
//         }

//         loc.left = w_xmin / total_score * IMAGE_WIDTH;
//         loc.top = w_ymin / total_score * IMAGE_HEIGHT;
//         loc.right = w_xmax / total_score * IMAGE_WIDTH;
//         loc.bottom = w_ymax / total_score * IMAGE_HEIGHT;
//       }

//       indexed_scores.clear();
//       indexed_scores.addAll(remained);
//       output_locs.add(loc);
//     }
//     return output_locs;
//   }

//   double _overlapSimilarity(Rect rect1, Rect rect2) {
//     bool intersects = rect1.intersects(rect2);
//     if (!intersects) return 0.0;
//     // if (!rect1.intersects(rect2)) return 0.0;

//     double intersection_area = rect1.intersection_area(rect2);
//     double normalization = rect1.height() * rect1.width() +
//         rect2.height() * rect2.width() -
//         intersection_area;

//     return normalization > 0.0 ? intersection_area / normalization : 0.0;
//   }

//   List<Anchor> _getAnchors(AnchorOption options) {
//     List<Anchor> _anchors = [];
//     if (options.stridesSize != options.numLayers) {
//       return [];
//     }
//     int layerID = 0;
//     while (layerID < options.stridesSize) {
//       List<double> anchorHeight = [];
//       List<double> anchorWidth = [];
//       List<double> aspectRatios = [];
//       List<double> scales = [];

//       int lastSameStrideLayer = layerID;
//       while (lastSameStrideLayer < options.stridesSize &&
//           options.strides[lastSameStrideLayer] == options.strides[layerID]) {
//         double scale = options.minScale +
//             (options.maxScale - options.minScale) *
//                 1.0 *
//                 lastSameStrideLayer /
//                 (options.stridesSize - 1.0);
//         if (lastSameStrideLayer == 0 && options.reduceBoxesInLowestLayer) {
//           aspectRatios.add(1.0);
//           aspectRatios.add(2.0);
//           aspectRatios.add(0.5);
//           scales.add(0.1);
//           scales.add(scale);
//           scales.add(scale);
//         } else {
//           for (int i = 0; i < options.aspectRatios.length; i++) {
//             aspectRatios.add(options.aspectRatios[i]);
//             scales.add(scale);
//           }

//           if (options.interpolatedScaleAspectRatio > 0.0) {
//             double scaleNext = 0.0;
//             if (lastSameStrideLayer == options.stridesSize - 1) {
//               scaleNext = 1.0;
//             } else {
//               scaleNext = options.minScale +
//                   (options.maxScale - options.minScale) *
//                       1.0 *
//                       (lastSameStrideLayer + 1) /
//                       (options.stridesSize - 1.0);
//             }
//             scales.add(sqrt(scale * scaleNext));
//             aspectRatios.add(options.interpolatedScaleAspectRatio);
//           }
//         }
//         lastSameStrideLayer++;
//       }
//       for (int i = 0; i < aspectRatios.length; i++) {
//         double ratioSQRT = sqrt(aspectRatios[i]);
//         anchorHeight.add(scales[i] / ratioSQRT);
//         anchorWidth.add(scales[i] * ratioSQRT);
//       }
//       int featureMapHeight = 0;
//       int featureMapWidth = 0;
//       if (options.featureMapHeightSize > 0) {
//         featureMapHeight = options.featureMapHeight[layerID];
//         featureMapWidth = options.featureMapWidth[layerID];
//       } else {
//         int stride = options.strides[layerID];
//         featureMapHeight = (1.0 * options.inputSizeHeight / stride).ceil();
//         featureMapWidth = (1.0 * options.inputSizeWidth / stride).ceil();
//       }

//       for (int y = 0; y < featureMapHeight; y++) {
//         for (int x = 0; x < featureMapWidth; x++) {
//           for (int anchorID = 0; anchorID < anchorHeight.length; anchorID++) {
//             double xCenter =
//                 (x + options.anchorOffsetX) * 1.0 / featureMapWidth;
//             double yCenter =
//                 (y + options.anchorOffsetY) * 1.0 / featureMapHeight;
//             double w = 0;
//             double h = 0;
//             if (options.fixedAnchorSize) {
//               w = 1.0;
//               h = 1.0;
//             } else {
//               w = anchorWidth[anchorID];
//               h = anchorHeight[anchorID];
//             }
//             _anchors.add(Anchor(xCenter, yCenter, h, w));
//           }
//         }
//       }
//       layerID = lastSameStrideLayer;
//     }
//     return _anchors;
//   }

//   void dispose() {
//     _interpreter?.close();
//   }
// }

// class ImagePickerHelper {
//   final _logger = Logger();

//   Future<XFile?> pickImageFromGallery() async {
//     final ImagePicker picker = ImagePicker();
//     final XFile? image = await picker.pickImage(source: ImageSource.gallery);
//     return image;
//   }

//   Future<Uint8List?> resizeImage(XFile pickedImage) async {
//     // Read image from the file.
//     Uint8List imageBytes = await pickedImage.readAsBytes();
//     img.Image? image = img.decodeImage(imageBytes);

//     // Resize the image to a 128x128 thumbnail (maintaining the aspect ratio).
//     img.Image thumbnail = img.copyResize(image!, width: 128, height: 128);

//     return thumbnail.getBytes();
//   }
// }
