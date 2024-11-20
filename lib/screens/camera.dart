import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:start/screens/create.dart';
import 'package:tflite/tflite.dart';
import 'dart:io';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Create(),
    );
  }
}

class CameraPage extends StatefulWidget {
  final Map<String, dynamic> userPreferences;
  const CameraPage({Key? key, required this.userPreferences}) : super(key: key);

  @override
  _CameraPageState createState() => _CameraPageState();
}

// Rest of the CameraPage implementation remains the same...
class _CameraPageState extends State<CameraPage> {
  File? _image;
  final ImagePicker _picker = ImagePicker();
  String _recognitionResult = "No results";
  String _detectionPrompt = "";
  String _segmentationPrompt = "";
  String _depthPrompt = "";
  bool _isProcessing = false; // New variable to track the processing state

  @override
  void initState() {
    super.initState();
    loadModels();
  }

  // Load all models in advance
  Future<void> loadModels() async {
    await Tflite.close();
    await Tflite.loadModel(
      model: "assets/ssd_mobilenet.tflite",
      labels: "assets/ssd_mobilenet_labels.txt",
    );
    await Tflite.loadModel(
      model: "assets/deeplabv3.tflite",
      labels: "assets/segmentation_labels.txt",
    );
    await Tflite.loadModel(
      model: "assets/midas.tflite",
    );
  }

  // Pick an image from either camera or gallery
  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _recognitionResult = "Image taken. Processing...";
        _isProcessing = true; // Set processing state
      });
      // Run each model in sequence and generate prompts
      processImage(_image!);
    } else {
      print('No image selected.');
    }
  }

  // Process the image with all three models sequentially
  Future<void> processImage(File image) async {
    await detectObjects(image); // First model
    await runSegmentation(image); // Second model
    await estimateDepth(image); // Third model

    setState(() {
      _recognitionResult = "Features extracted. Prompts generated.";
      _isProcessing = false; // Reset processing state
    });
  }

  Future<void> detectObjects(File image) async {
    var recognitions = await Tflite.detectObjectOnImage(
      path: image.path,
      model: "SSD MobileNet",
      numResultsPerClass: 3,
      threshold: 0.5,
    );

    setState(() {
      if (recognitions != null && recognitions.isNotEmpty) {
        _detectionPrompt = "Detected objects: " + recognitions.toString();
      } else {
        _detectionPrompt = "No objects detected.";
      }
    });
    print("Object Detection Results: $_detectionPrompt");
  }

  Future<void> runSegmentation(File image) async {
    var recognitions = await Tflite.runSegmentationOnImage(
      path: image.path,
    );

    setState(() {
      if (recognitions != null) {
        _segmentationPrompt = "Segmentation completed.";
      } else {
        _segmentationPrompt = "Segmentation failed.";
      }
    });
    print("Segmentation Results: $_segmentationPrompt");
  }

  Future<void> estimateDepth(File image) async {
    var recognitions = await Tflite.runModelOnImage(
      path: image.path,
    );

    setState(() {
      if (recognitions != null) {
        _depthPrompt = "Depth estimation completed.";
      } else {
        _depthPrompt = "Depth estimation failed.";
      }
    });
    print("Depth Estimation Results: $_depthPrompt");
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Automated Feature Extraction Example')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image == null
                ? Text('No image selected.')
                : _isProcessing
                    ? Column(
                        children: [
                          Text('Image taken.'),
                          CircularProgressIndicator(),
                        ],
                      )
                    : Image.file(_image!),
            SizedBox(height: 20),
            Text(_recognitionResult),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => _pickImage(ImageSource.camera),
              child: Text('Pick Image from Camera'),
            ),
            ElevatedButton(
              onPressed: () => _pickImage(ImageSource.gallery),
              child: Text('Pick Image from Gallery'),
            ),
            SizedBox(height: 20),
            Text("Detection Prompt: $_detectionPrompt"),
            Text("Segmentation Prompt: $_segmentationPrompt"),
            Text("Depth Prompt: $_depthPrompt"),
          ],
        ),
      ),
    );
  }
}
