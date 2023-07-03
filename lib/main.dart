import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mediapipe/logic/tflite_helper.dart';

import 'logic/facedetection_helper.dart';

late RootIsolateToken rootIsolateToken;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  rootIsolateToken = RootIsolateToken.instance!;

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.amber,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late Future _future;

  XFile? pickedImage;
  Uint8List? resizedImage;
  int detections = 0;

  @override
  void initState() {
    super.initState();
    _future = TFLiteHelper().initialiseModel();
  }

  @override
  void dispose() {
    super.dispose();
    TFLiteHelper().dispose();
  }

  void _pickImageFromGallery() async {
    pickedImage = null;
    resizedImage = null;
    detections = 0;

    pickedImage = await ImagePickerHelper().pickImageFromGallery();

    if (pickedImage == null) return;

    // resize the image to 128x128
    resizedImage = await ImagePickerHelper().resizeImage(pickedImage!);

    detections = await TFLiteHelper().detectFace(
      rootIsolateToken,
      resizedImage!,
    );

    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Mediapipe"),
      ),
      body: FutureBuilder(
          future: _future,
          builder: (context, snapshot) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: <Widget>[
                    if (pickedImage != null)
                      SizedBox(
                        width: 256,
                        height: 256,
                        child: Image.file(
                          File(pickedImage!.path),
                          errorBuilder: (BuildContext context, Object error,
                              StackTrace? stackTrace) {
                            return const Center(
                              child: Text('This image type is not supported'),
                            );
                          },
                        ),
                      ),
                    const SizedBox(height: 16.0),
                    Text("Detections: $detections faces."),
                  ],
                ),
              ),
            );
          }),
      floatingActionButton: FloatingActionButton(
        onPressed: _pickImageFromGallery,
        tooltip: 'Pick Image',
        child: const Icon(Icons.add_a_photo_outlined),
      ),
    );
  }
}
