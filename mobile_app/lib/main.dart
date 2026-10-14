import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    final cameras = await availableCameras();

    runApp(SignEchoApp(cameras: cameras));
  } catch (e) {
    debugPrint('CAMERA INITIALIZATION ERROR: $e');

    runApp(
      const SignEchoApp(
        cameras: [],
      ),
    );
  }
}

// ============================================================
// APP
// ============================================================

class SignEchoApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const SignEchoApp({
    super.key,
    required this.cameras,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'SignEcho',
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: SignEchoHome(
        cameras: cameras,
      ),
    );
  }
}

// ============================================================
// HOME SCREEN
// ============================================================

class SignEchoHome extends StatefulWidget {
  final List<CameraDescription> cameras;

  const SignEchoHome({
    super.key,
    required this.cameras,
  });

  @override
  State<SignEchoHome> createState() => _SignEchoHomeState();
}

class _SignEchoHomeState extends State<SignEchoHome> {
  // ==========================================================
  // OBJECTS
  // ==========================================================

  CameraController? _cameraController;

  HandLandmarkerPlugin? _handLandmarker;

  Interpreter? _interpreter;

  StreamSubscription<List<Hand>>? _landmarkSubscription;

  // ==========================================================
  // STATES
  // ==========================================================

  bool _cameraStarted = false;

  bool _modelLoaded = false;

  bool _processing = false;

  String _status = 'Loading model...';

  String _prediction = 'Waiting...';

  List<String> _labels = [];

  // ==========================================================
  // DYNAMIC SIGN SEQUENCE
  // ==========================================================

final List<List<double>> _sequence = [];

static const int sequenceLength = 30;
static const int featureLength = 63;

// Prevent continuous predictions
bool _predictionRunning = false;

// Small pause between gestures
bool _gestureCooldown = false;

  // ==========================================================
  // INIT
  // ==========================================================

  @override
  void initState() {
    super.initState();

    _loadModel();
  }

  // ==========================================================
  // LOAD TFLITE MODEL
  // ==========================================================

  Future<void> _loadModel() async {
    try {
      debugPrint('Loading TFLite model...');

      _interpreter = await Interpreter.fromAsset(
        'assets/models/dynamic_sign_model.tflite',
      );

      debugPrint('TFLite model loaded successfully');

      // Load labels without using BuildContext after await.
      final labelText = await rootBundle.loadString(
        'assets/labels.txt',
      );

      _labels = labelText
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();

      debugPrint('Labels: $_labels');

      debugPrint(
        'Number of labels: ${_labels.length}',
      );

      if (!mounted) return;

      setState(() {
        _modelLoaded = true;
        _status = 'Model ready';
      });
    } catch (e) {
      debugPrint('MODEL ERROR: $e');

      if (!mounted) return;

      setState(() {
        _modelLoaded = false;
        _status = 'Model loading failed';
      });
    }
  }

  // ==========================================================
  // START DETECTION
  // ==========================================================

  Future<void> _startDetection() async {
    if (!_modelLoaded || _interpreter == null) {
      if (!mounted) return;

      setState(() {
        _status = 'Model is not ready';
      });

      return;
    }

    if (widget.cameras.isEmpty) {
      if (!mounted) return;

      setState(() {
        _status = 'No camera found';
      });

      return;
    }

    final camera = widget.cameras.firstWhere(
      (camera) =>
          camera.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras.first,
    );

    try {
      debugPrint('Creating Hand Landmarker...');

      _handLandmarker = HandLandmarkerPlugin.create(
        numHands: 1,
        minHandDetectionConfidence: 0.7,
        delegate: HandLandmarkerDelegate.gpu,
      );

      debugPrint('Hand Landmarker created');

      _cameraController = CameraController(
        camera,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      debugPrint('Initializing camera...');

      await _cameraController!.initialize();

      debugPrint('Camera initialized');

      // Listen for MediaPipe landmark results.
      _landmarkSubscription =
          _handLandmarker!.landmarkStream.listen(
        _onLandmarks,
        onError: (error) {
          debugPrint(
            'LANDMARK STREAM ERROR: $error',
          );
        },
      );

      debugPrint('Starting detection...');

      // Mark detection as started BEFORE receiving camera frames.
      if (!mounted) return;

      setState(() {
        _cameraStarted = true;
        _status = 'Show a sign...';
        _prediction = 'Waiting...';
      });

      await _cameraController!.startImageStream(
        _processCameraImage,
      );

      debugPrint('Camera image stream started');
    } catch (e) {
      debugPrint(
        'CAMERA/HAND LANDMARK ERROR: $e',
      );

      try {
        await _cameraController?.dispose();
      } catch (_) {}

      _cameraController = null;

      try {
        _handLandmarker?.dispose();
      } catch (_) {}

      _handLandmarker = null;

      if (!mounted) return;

      setState(() {
        _cameraStarted = false;
        _status = 'Camera error';
      });
    }
  }

  // ==========================================================
  // CAMERA FRAME → MEDIAPIPE
  // ==========================================================

  void _processCameraImage(CameraImage image) {
    if (!_cameraStarted &&
        _cameraController == null) {
      return;
    }

    if (_handLandmarker == null ||
        _cameraController == null) {
      return;
    }

    if (_processing) {
      return;
    }

    _processing = true;

    try {
      _handLandmarker!.processFrame(
        image,
        _cameraController!.description.sensorOrientation,
      );
    } catch (e) {
      debugPrint(
        'FRAME PROCESSING ERROR: $e',
      );
    } finally {
      _processing = false;
    }
  }

  // ==========================================================
  // RECEIVE HAND LANDMARKS
  // ==========================================================

 void _onLandmarks(List<Hand> hands) {
  if (!_cameraStarted) {
    return;
  }

  // --------------------------------------------------------
  // DO NOT COLLECT FRAMES WHILE PREDICTING OR IN COOLDOWN
  // --------------------------------------------------------

  if (_predictionRunning || _gestureCooldown) {
    return;
  }

  // --------------------------------------------------------
  // NO HAND
  // --------------------------------------------------------

  // --------------------------------------------------------
// NO HAND
// --------------------------------------------------------

if (hands.isEmpty) {

  // If we are waiting for the user to remove
  // the previous gesture, allow next gesture.
  if (_gestureCooldown) {
    _gestureCooldown = false;
    _sequence.clear();

    if (mounted) {
      setState(() {
        _status = 'Show a sign...';
      });
    }

    return;
  }

  if (mounted) {
    setState(() {
      _status = 'No hand detected';
    });
  }

  return;
}

  // --------------------------------------------------------
  // FIRST HAND
  // --------------------------------------------------------

  final hand = hands.first;

  if (hand.landmarks.length != 21) {
    return;
  }

  // --------------------------------------------------------
  // NORMALIZE
  // --------------------------------------------------------

  final features = _normalizeLandmarks(
    hand.landmarks,
  );

  if (features.length != featureLength) {
    return;
  }

  // --------------------------------------------------------
  // ADD ONE FRAME
  // --------------------------------------------------------

  _sequence.add(features);

  // --------------------------------------------------------
  // SHOW FRAME COUNT
  // --------------------------------------------------------

  if (mounted) {
    setState(() {
      _status =
          'Collecting frames: '
          '${_sequence.length}/$sequenceLength';
    });
  }

  // --------------------------------------------------------
  // WAIT UNTIL 30 FRAMES
  // --------------------------------------------------------

  if (_sequence.length < sequenceLength) {
    return;
  }

  // --------------------------------------------------------
  // STOP COLLECTING
  // --------------------------------------------------------

  _predictionRunning = true;

  // Make a COPY of exactly 30 frames.
  final sequenceToPredict = _sequence
      .map((frame) => List<double>.from(frame))
      .toList();

  // Clear sequence immediately.
  _sequence.clear();

  if (mounted) {
    setState(() {
      _status = 'Recognizing gesture...';
    });
  }

  // --------------------------------------------------------
  // PREDICT THE COPIED 30 FRAMES
  // --------------------------------------------------------

  _runPrediction(sequenceToPredict);
}

  // ==========================================================
  // NORMALIZE LANDMARKS
  //
  // MUST MATCH PYTHON DATA COLLECTION
  // ==========================================================

  List<double> _normalizeLandmarks(
    List<Landmark> landmarks,
  ) {
    if (landmarks.length != 21) {
      return [];
    }

    // Wrist = landmark 0
    final wrist = landmarks[0];

    final values = <double>[];

    // --------------------------------------------------------
    // SUBTRACT WRIST
    // --------------------------------------------------------

    for (final landmark in landmarks) {
      final x = landmark.x - wrist.x;
      final y = landmark.y - wrist.y;
      final z = landmark.z - wrist.z;

      values.add(x);
      values.add(y);
      values.add(z);
}

    // We should have 21 × 3 = 63 features.
    if (values.length != featureLength) {
      return [];
    }

    // --------------------------------------------------------
    // SCALE NORMALIZATION
    // --------------------------------------------------------

    double maxValue = 0;

    for (final value in values) {
      maxValue = math.max(
        maxValue,
        value.abs(),
      );
    }

    if (maxValue > 0) {
      for (int i = 0; i < values.length; i++) {
        values[i] = values[i] / maxValue;
      }
    }

    return values;
  }

  // ==========================================================
  // TFLITE PREDICTION
  // ==========================================================

  void _runPrediction(
  List<List<double>> sequenceToPredict,
) {
  if (_interpreter == null) {
    _predictionRunning = false;
    return;
  }

  if (sequenceToPredict.length != sequenceLength) {
    _predictionRunning = false;
    return;
  }

  if (_labels.isEmpty) {
    debugPrint('No labels loaded');

    _predictionRunning = false;
    return;
  }

    try {
    // ------------------------------------------------------
    // INPUT SHAPE
    //
    // [1, 30, 63]
    // ------------------------------------------------------

    final input = <List<List<double>>>[
      sequenceToPredict
          .map(
            (frame) => List<double>.from(frame),
          )
          .toList(),
    ];

    // ------------------------------------------------------
    // DEBUG INPUT
    // ------------------------------------------------------

    debugPrint(
      'INPUT SHAPE: '
      '[${input.length}, '
      '${input[0].length}, '
      '${input[0][0].length}]',
    );

    debugPrint(
      'FIRST FRAME: ${input[0][0]}',
    );

    debugPrint(
      'LAST FRAME: ${input[0][29]}',
    );

    // ------------------------------------------------------
    // OUTPUT
    // ------------------------------------------------------

    final output = List.generate(
      1,
      (_) => List<double>.filled(
        _labels.length,
        0,
      ),
    );

    // ------------------------------------------------------
    // RUN MODEL
    // ------------------------------------------------------

    _interpreter!.run(
      input,
      output,
    );

    final scores = output[0];

    if (scores.isEmpty) {
      _predictionRunning = false;
      return;
    }

    // ------------------------------------------------------
    // FIND BEST CLASS
    // ------------------------------------------------------

    int bestIndex = 0;
    double bestScore = scores[0];

    for (int i = 1; i < scores.length; i++) {
      if (scores[i] > bestScore) {
        bestScore = scores[i];
        bestIndex = i;
      }
    }

    // ------------------------------------------------------
    // GET LABEL
    // ------------------------------------------------------

    if (bestIndex < _labels.length) {
      final detectedSign = _labels[bestIndex];

      final confidence = bestScore * 100;

      if (mounted) {
        setState(() {
          _prediction = detectedSign;

          _status =
              'Detected with '
              '${confidence.toStringAsFixed(1)}% confidence';
        });
      }

      debugPrint('================================');
      debugPrint('PREDICTION: $detectedSign');
      debugPrint(
        'CONFIDENCE: '
        '${confidence.toStringAsFixed(1)}%',
      );
      debugPrint('CLASS INDEX: $bestIndex');
      debugPrint('================================');
    }

    // ------------------------------------------------------
    // PREDICTION FINISHED
    // ------------------------------------------------------

    _predictionRunning = false;

    // ------------------------------------------------------
    // COOLDOWN
    // ------------------------------------------------------

    _gestureCooldown = true;

    if (mounted) {
      setState(() {
        _status =
            'Gesture detected. Show next sign...';
      });
    }

    Future.delayed(
      const Duration(milliseconds: 1200),
      () {
        _gestureCooldown = false;

        if (mounted) {
          setState(() {
            _status = 'Show a sign...';
          });
        }
      },
    );
  } catch (e) {
    debugPrint(
      'TFLITE PREDICTION ERROR: $e',
    );

    _predictionRunning = false;

    if (mounted) {
      setState(() {
        _status = 'Prediction error';
      });
    }
  }
}

  // ==========================================================
  // STOP DETECTION
  // ==========================================================

  Future<void> _stopDetection() async {
    try {
      await _cameraController?.stopImageStream();
    } catch (_) {}

    try {
      await _landmarkSubscription?.cancel();
    } catch (_) {}

    _landmarkSubscription = null;

    try {
      await _cameraController?.dispose();
    } catch (_) {}

    _cameraController = null;

    try {
      _handLandmarker?.dispose();
    } catch (_) {}

    _handLandmarker = null;

    _sequence.clear();

    _processing = false;

    if (!mounted) return;

    setState(() {
      _cameraStarted = false;

      _prediction = 'Waiting...';

      _status = 'Detection stopped';
    });
  }

  // ==========================================================
  // DISPOSE
  // ==========================================================

  @override
  void dispose() {
    _landmarkSubscription?.cancel();

    _cameraController?.dispose();

    _handLandmarker?.dispose();

    _interpreter?.close();

    super.dispose();
  }

  // ==========================================================
  // UI
  // ==========================================================

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'SignEcho',
          style: TextStyle(
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: true,
      ),

      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              const SizedBox(height: 10),

              // =================================================
              // CAMERA PREVIEW
              // =================================================

              Expanded(
                child: Container(
                  width: double.infinity,
                  clipBehavior: Clip.hardEdge,
                  decoration: BoxDecoration(
                    borderRadius:
                        BorderRadius.circular(24),
                    color: Colors.black,
                    border: Border.all(
                      color: Colors.white24,
                      width: 2,
                    ),
                  ),

                  child: _cameraStarted &&
                          controller != null &&
                          controller.value.isInitialized
                      ? CameraPreview(controller)
                      : const Center(
                          child: Column(
                            mainAxisAlignment:
                                MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.camera_alt,
                                size: 70,
                                color: Colors.white54,
                              ),

                              SizedBox(height: 15),

                              Text(
                                'Camera Preview',
                                style: TextStyle(
                                  fontSize: 20,
                                  color: Colors.white70,
                                ),
                              ),
                            ],
                          ),
                        ),
                ),
              ),

              const SizedBox(height: 20),

              // =================================================
              // DETECTED SIGN
              // =================================================

              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),

                decoration: BoxDecoration(
                  borderRadius:
                      BorderRadius.circular(20),
                  color: Colors.white10,
                ),

                child: Column(
                  children: [
                    const Text(
                      'Detected Sign',
                      style: TextStyle(
                        fontSize: 16,
                        color: Colors.white60,
                      ),
                    ),

                    const SizedBox(height: 8),

                    Text(
                      _prediction,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                      ),
                    ),

                    const SizedBox(height: 8),

                    Text(
                      _status,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 14,
                        color: Colors.white60,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 20),

              // =================================================
              // START / STOP BUTTON
              // =================================================

              SizedBox(
                width: double.infinity,
                height: 55,

                child: ElevatedButton.icon(
                  onPressed: _cameraStarted
                      ? _stopDetection
                      : _startDetection,

                  icon: Icon(
                    _cameraStarted
                        ? Icons.stop
                        : Icons.camera,
                  ),

                  label: Text(
                    _cameraStarted
                        ? 'STOP DETECTION'
                        : 'START SIGN DETECTION',

                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),

              const SizedBox(height: 15),
            ],
          ),
        ),
      ),
    );
  }
}