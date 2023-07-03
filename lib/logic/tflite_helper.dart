import 'dart:isolate';

import 'package:flutter/services.dart';
import 'package:logger/logger.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

import 'facedetection_helper.dart';

class TFLiteHelper {
  static tfl.Interpreter? _interpreter;
  final _logger = Logger();

  Future<tfl.Interpreter?> initialiseModel() async {
    try {
      final tfl.Interpreter interpreter =
          await tfl.Interpreter.fromAsset('assets/models/blazeface.tflite');

      _interpreter = interpreter;

      return interpreter;
    } on PlatformException {
      _logger.e('Failed to load model.');
    } catch (e, s) {
      _logger.e(e);
      _logger.e(s);
    }

    return null;
  }

  Future<int> detectFace(
    final RootIsolateToken rootIsolateToken,
    Uint8List thumbnail,
  ) async {
    // Guard clause
    if (_interpreter == null) return -1;

    // TODO(KM): why is _interpreter null in isolate and not otherwise.
    // int faces = 0;
    // try {
    //   faces = await Isolate.run<int>(
    //     () => FaceDetectionHelper.detectFace(
    //       rootIsolateToken,
    //       thumbnail,
    //       _interpreter,
    //     ),
    //   );
    //   return faces;
    // } catch (e, s) {
    //   _logger.e("Error occurred while spawing an Isolate");
    //   _logger.e(e);
    //   _logger.e(s);
    //   return -1;
    // }

    return FaceDetectionHelper.detectFace(
      rootIsolateToken,
      thumbnail,
      _interpreter!,
    );
  }

  void dispose() {
    _interpreter?.close();
  }
}
