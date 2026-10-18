import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    final cameras = await availableCameras();
    runApp(SignEchoApp(cameras: cameras));
  } catch (e) {
    debugPrint('CAMERA INITIALIZATION ERROR: $e');
    runApp(const SignEchoApp(cameras: []));
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
        scaffoldBackgroundColor: const Color(0xFF080B18),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF8B5CF6),
          brightness: Brightness.dark,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF080B18),
          foregroundColor: Colors.white,
          elevation: 0,
          centerTitle: false,
        ),
      ),
      home: SignEchoHome(cameras: cameras),
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
  CameraController? _cameraController;
  HandLandmarkerPlugin? _handLandmarker;
  Interpreter? _interpreter;
  StreamSubscription<List<Hand>>? _landmarkSubscription;

  final FlutterTts _tts = FlutterTts();
  bool _ttsReady = false;

  static const List<String> _expectedLabels = [
    'Bye',
    'drink',
    'emergency',
    'hello',
    'hungry',
    'no',
    'phone',
    'sorry',
    'thankyou',
    'toilets',
    'where',
    'yes',
  ];

  static const double _minimumConfidence = 0.70;
  static const double _minimumMargin = 0.15;

  bool _cameraStarted = false;
  bool _modelLoaded = false;
  bool _processing = false;
  bool _predictionRunning = false;
  bool _gestureCooldown = false;

  String _status = 'Loading model...';
  String _prediction = 'Waiting...';

  List<String> _labels = [];

  final List<List<double>> _sequence = [];

  static const int sequenceLength = 100;
  static const int featureLength = 63;

  // ============================================================
  // INITIALIZATION
  // ============================================================

  @override
  void initState() {
    super.initState();
    _initTts();
    _loadModel();
  }

  Future<void> _initTts() async {
    try {
      await _tts.setLanguage('en-US');
      await _tts.setSpeechRate(0.45);
      await _tts.setVolume(1.0);
      await _tts.setPitch(1.0);

      _ttsReady = true;
      debugPrint('Text-to-speech initialized');
    } catch (e) {
      debugPrint('TTS INITIALIZATION ERROR: $e');
      _ttsReady = false;
    }
  }

  Future<void> _speakSign(String sign) async {
    if (!_ttsReady) {
      debugPrint('TTS is not ready; cannot speak: $sign');
      return;
    }

    try {
      await _tts.stop();
      await _tts.speak(sign);
    } catch (e) {
      debugPrint('TTS SPEAK ERROR: $e');
    }
  }

  // ============================================================
  // LOAD TFLITE MODEL
  // ============================================================

  Future<void> _loadModel() async {
    try {
      debugPrint('Loading TFLite model...');

      _interpreter = await Interpreter.fromAsset(
        'assets/models/dynamic_sign_model.tflite',
      );

      debugPrint('TFLite model loaded successfully');

      final labelText = await rootBundle.loadString(
        'assets/labels.txt',
      );

      _labels = labelText
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();

      debugPrint('Labels: $_labels');
      debugPrint('Number of labels: ${_labels.length}');

      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;

      debugPrint('TFLite input shape: $inputShape');
      debugPrint('TFLite output shape: $outputShape');

      if (inputShape.length != 3 ||
          inputShape[0] != 1 ||
          inputShape[1] != sequenceLength ||
          inputShape[2] != featureLength) {
        throw StateError(
          'Unexpected model input shape $inputShape; '
          'expected [1, 100, 63].',
        );
      }

      if (outputShape.length != 2 ||
          outputShape[0] != 1 ||
          outputShape[1] != _expectedLabels.length ||
          _labels.length != _expectedLabels.length ||
          !_labels.asMap().entries.every(
            (entry) => entry.value == _expectedLabels[entry.key],
          )) {
        throw StateError(
          'Model output/labels do not match the expected 12 trained classes.',
        );
      }

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

  // ============================================================
  // START DETECTION
  // ============================================================

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

      _landmarkSubscription =
          _handLandmarker!.landmarkStream.listen(
        _onLandmarks,
        onError: (error) {
          debugPrint('LANDMARK STREAM ERROR: $error');
        },
      );

      debugPrint('Starting detection...');

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
      debugPrint('CAMERA/HAND LANDMARK ERROR: $e');

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

  // ============================================================
  // CAMERA FRAME -> MEDIAPIPE
  // ============================================================

  void _processCameraImage(CameraImage image) {
    if (!_cameraStarted && _cameraController == null) {
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
      debugPrint('FRAME PROCESSING ERROR: $e');
    } finally {
      _processing = false;
    }
  }

  // ============================================================
  // RECEIVE HAND LANDMARKS
  // ============================================================

  void _onLandmarks(List<Hand> hands) {
    if (!_cameraStarted) {
      return;
    }

    if (_predictionRunning || _gestureCooldown) {
      return;
    }

    if (hands.isEmpty) {
      if (mounted) {
        setState(() {
          _status = 'No hand detected';
        });
      }
      return;
    }

    final hand = hands.first;

    if (hand.landmarks.length != 21) {
      return;
    }

    final features = _normalizeLandmarks(
      hand.landmarks,
    );

    if (features.length != featureLength) {
      return;
    }

    _sequence.add(features);

    if (mounted) {
      setState(() {
        _status =
            'Collecting frames: '
            '${_sequence.length}/$sequenceLength';
      });
    }

    if (_sequence.length < sequenceLength) {
      return;
    }

    _predictionRunning = true;

    final sequenceToPredict = _sequence
        .map((frame) => List<double>.from(frame))
        .toList();

    _sequence.clear();

    if (mounted) {
      setState(() {
        _status = 'Recognizing gesture...';
      });
    }

    _runPrediction(sequenceToPredict);
  }

  // ============================================================
  // NORMALIZE LANDMARKS
  // Must match Python preprocessing
  // ============================================================

  List<double> _normalizeLandmarks(
    List<Landmark> landmarks,
  ) {
    if (landmarks.length != 21) {
      return [];
    }

    final wrist = landmarks[0];
    final values = <double>[];

    for (final landmark in landmarks) {
      final x = landmark.x - wrist.x;
      final y = landmark.y - wrist.y;
      final z = landmark.z - wrist.z;

      values.add(x);
      values.add(y);
      values.add(z);
    }

    if (values.length != featureLength) {
      return [];
    }

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

  // ============================================================
  // TFLITE PREDICTION
  // ============================================================

  void _runPrediction(
    List<List<double>> sequenceToPredict,
  ) {
    if (_interpreter == null ||
        sequenceToPredict.length != sequenceLength ||
        _labels.length != _expectedLabels.length) {
      _predictionRunning = false;
      return;
    }

    try {
      final input = <List<List<double>>>[
        sequenceToPredict
            .map((frame) => List<double>.from(frame))
            .toList(),
      ];

      debugPrint(
        'INPUT SHAPE: '
        '[${input.length}, ${input[0].length}, '
        '${input[0][0].length}]',
      );

      debugPrint('FIRST FRAME: ${input[0][0]}');
      debugPrint('LAST FRAME: ${input[0][99]}');

      final output = List.generate(
        1,
        (_) => List<double>.filled(_labels.length, 0.0),
      );

      _interpreter!.run(input, output);

      final scores = output[0];

      if (scores.length != _labels.length ||
          scores.isEmpty) {
        throw StateError(
          'Model output count does not match labels.',
        );
      }

      int bestIndex = 0;
      double bestScore = scores[0];
      double secondBestScore = double.negativeInfinity;

      for (int i = 1; i < scores.length; i++) {
        if (scores[i] > bestScore) {
          secondBestScore = bestScore;
          bestScore = scores[i];
          bestIndex = i;
        } else if (scores[i] > secondBestScore) {
          secondBestScore = scores[i];
        }
      }

      final margin = bestScore - secondBestScore;
      final confidencePercent = bestScore * 100;
      final detectedSign = _labels[bestIndex];

      final accepted =
          bestScore >= _minimumConfidence &&
          margin >= _minimumMargin &&
          bestScore <= 1.0;

      debugPrint('================================');
      debugPrint('TOP LABEL: $detectedSign');
      debugPrint('TOP SCORE: ${bestScore.toStringAsFixed(4)}');
      debugPrint('TOP-TWO MARGIN: ${margin.toStringAsFixed(4)}');
      debugPrint('CLASS INDEX: $bestIndex');
      debugPrint('ACCEPTED: $accepted');
      debugPrint('================================');

      if (mounted) {
        setState(() {
          if (accepted) {
            _prediction = detectedSign;
            _status =
                'Recognized: '
                '${confidencePercent.toStringAsFixed(1)}% confidence';
          } else {
            _prediction = 'Uncertain';
            _status =
                'Sign unclear. Try again with your hand clearly visible.';
          }
        });
      }

      if (accepted) {
        unawaited(_speakSign(detectedSign));
      }

      _predictionRunning = false;
      _gestureCooldown = true;

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
      debugPrint('TFLITE PREDICTION ERROR: $e');

      _predictionRunning = false;

      if (mounted) {
        setState(() {
          _prediction = 'Try again';
          _status = 'Prediction error';
        });
      }
    }
  }

  // ============================================================
  // STOP DETECTION
  // ============================================================

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
    _predictionRunning = false;
    _gestureCooldown = false;

    if (!mounted) return;

    setState(() {
      _cameraStarted = false;
      _prediction = 'Waiting...';
      _status = 'Detection stopped';
    });
  }

  // ============================================================
  // DISPOSE
  // ============================================================

  @override
  void dispose() {
    _landmarkSubscription?.cancel();
    _cameraController?.dispose();
    _handLandmarker?.dispose();
    _interpreter?.close();
    _tts.stop();

    super.dispose();
  }

  // ============================================================
  // INFO CARD
  // ============================================================

  Widget _infoCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color accent,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF12172A),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.07),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: accent.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(
              icon,
              color: accent,
              size: 22,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: const TextStyle(
                    color: Color(0xFF9CA3B8),
                    fontSize: 11,
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ============================================================
  // STATUS PILL
  // ============================================================

  Widget _statusPill() {
    final Color statusColor = _cameraStarted
        ? const Color(0xFF34D399)
        : (_modelLoaded
            ? const Color(0xFF60A5FA)
            : const Color(0xFFFBBF24));

    final String statusText = _cameraStarted
        ? 'LIVE DETECTION'
        : (_modelLoaded ? 'MODEL READY' : 'INITIALIZING');

    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 11,
        vertical: 7,
      ),
      decoration: BoxDecoration(
        color: statusColor.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(
          color: statusColor.withValues(alpha: 0.35),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 7,
            height: 7,
            decoration: BoxDecoration(
              color: statusColor,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 7),
          Text(
            statusText,
            style: TextStyle(
              color: statusColor,
              fontSize: 10,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.8,
            ),
          ),
        ],
      ),
    );
  }

  // ============================================================
  // UI
  // ============================================================

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;

    final bool cameraReady =
        _cameraStarted &&
        controller != null &&
        controller.value.isInitialized;

    final double progress =
        (_sequence.length / sequenceLength)
            .clamp(0.0, 1.0)
            .toDouble();

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF10142B),
              Color(0xFF080B18),
              Color(0xFF101827),
            ],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(
              20,
              14,
              20,
              28,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Brand header
                Row(
                  children: [
                    Container(
                      width: 48,
                      height: 48,
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [
                            Color(0xFF8B5CF6),
                            Color(0xFF22D3EE),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(16),
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xFF8B5CF6)
                                .withValues(alpha: 0.25),
                            blurRadius: 18,
                            offset: const Offset(0, 6),
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.sign_language_rounded,
                        color: Colors.white,
                        size: 27,
                      ),
                    ),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'SignEcho',
                            style: TextStyle(
                              fontSize: 23,
                              fontWeight: FontWeight.w800,
                              letterSpacing: -0.6,
                              color: Colors.white,
                            ),
                          ),
                          SizedBox(height: 2),
                          Text(
                            'SIGN. CONNECT. COMMUNICATE.',
                            style: TextStyle(
                              fontSize: 9,
                              letterSpacing: 1.35,
                              fontWeight: FontWeight.w700,
                              color: Color(0xFF9CA3B8),
                            ),
                          ),
                        ],
                      ),
                    ),
                    _statusPill(),
                  ],
                ),

                const SizedBox(height: 28),

                // Intro copy
                const Text(
                  'Every gesture\nhas a voice.',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 33,
                    height: 1.12,
                    fontWeight: FontWeight.w800,
                    letterSpacing: -1.0,
                  ),
                ),

                const SizedBox(height: 10),

                const Text(
                  'Turn sign language into meaningful communication\n'
                  'with real-time AI-powered recognition.',
                  style: TextStyle(
                    color: Color(0xFFADB5CA),
                    fontSize: 13,
                    height: 1.55,
                  ),
                ),

                const SizedBox(height: 23),

                // Camera section title
                Row(
                  children: [
                    const Expanded(
                      child: Text(
                        'LIVE RECOGNITION',
                        style: TextStyle(
                          color: Color(0xFFE4E7F2),
                          fontWeight: FontWeight.w800,
                          fontSize: 11,
                          letterSpacing: 1.4,
                        ),
                      ),
                    ),
                    Icon(
                      Icons.auto_awesome_rounded,
                      color: const Color(0xFF22D3EE)
                          .withValues(alpha: 0.9),
                      size: 17,
                    ),
                    const SizedBox(width: 6),
                    const Text(
                      'AI VISION',
                      style: TextStyle(
                        color: Color(0xFF22D3EE),
                        fontSize: 10,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 0.7,
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 12),

                // Camera preview card
                Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: const Color(0xFF111629),
                    borderRadius: BorderRadius.circular(25),
                    border: Border.all(
                      color: _cameraStarted
                          ? const Color(0xFF22D3EE)
                              .withValues(alpha: 0.55)
                          : Colors.white.withValues(alpha: 0.10),
                      width: 1.2,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xFF050711)
                            .withValues(alpha: 0.28),
                        blurRadius: 25,
                        offset: const Offset(0, 12),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      ClipRRect(
                        borderRadius: const BorderRadius.vertical(
                          top: Radius.circular(24),
                        ),
                        child: AspectRatio(
                          aspectRatio: 4 / 3,
                          child: cameraReady
                              ? CameraPreview(controller)
                              : Container(
                                  decoration: const BoxDecoration(
                                    gradient: LinearGradient(
                                      begin: Alignment.topLeft,
                                      end: Alignment.bottomRight,
                                      colors: [
                                        Color(0xFF1C2340),
                                        Color(0xFF101525),
                                      ],
                                    ),
                                  ),
                                  child: Center(
                                    child: Column(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Container(
                                          width: 76,
                                          height: 76,
                                          decoration: BoxDecoration(
                                            color: const Color(0xFF8B5CF6)
                                                .withValues(alpha: 0.15),
                                            shape: BoxShape.circle,
                                            border: Border.all(
                                              color: const Color(0xFF8B5CF6)
                                                  .withValues(alpha: 0.35),
                                            ),
                                          ),
                                          child: const Icon(
                                            Icons.videocam_rounded,
                                            color: Color(0xFFB9A0FF),
                                            size: 35,
                                          ),
                                        ),
                                        const SizedBox(height: 14),
                                        const Text(
                                          'Your camera preview appears here',
                                          style: TextStyle(
                                            color: Color(0xFFE5E7F2),
                                            fontSize: 13,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                        const SizedBox(height: 6),
                                        Text(
                                          _modelLoaded
                                              ? 'Ready when you are'
                                              : 'Loading recognition model…',
                                          style: const TextStyle(
                                            color: Color(0xFF9CA3B8),
                                            fontSize: 11,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                        ),
                      ),

                      Padding(
                        padding: const EdgeInsets.fromLTRB(
                          15,
                          13,
                          15,
                          14,
                        ),
                        child: Column(
                          children: [
                            Row(
                              children: [
                                Icon(
                                  cameraReady
                                      ? Icons.center_focus_strong_rounded
                                      : Icons.info_outline_rounded,
                                  size: 17,
                                  color: cameraReady
                                      ? const Color(0xFF34D399)
                                      : const Color(0xFF9CA3B8),
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Text(
                                    _status,
                                    maxLines: 2,
                                    overflow: TextOverflow.ellipsis,
                                    style: const TextStyle(
                                      color: Color(0xFFD7DCEC),
                                      fontSize: 11,
                                      height: 1.3,
                                    ),
                                  ),
                                ),
                                Text(
                                  '${_sequence.length}/$sequenceLength',
                                  style: const TextStyle(
                                    color: Color(0xFF22D3EE),
                                    fontSize: 11,
                                    fontWeight: FontWeight.w800,
                                  ),
                                ),
                              ],
                            ),

                            const SizedBox(height: 10),

                            ClipRRect(
                              borderRadius: BorderRadius.circular(20),
                              child: LinearProgressIndicator(
                                value: cameraReady ? progress : 0,
                                minHeight: 5,
                                backgroundColor: Colors.white
                                    .withValues(alpha: 0.08),
                                valueColor:
                                    const AlwaysStoppedAnimation<Color>(
                                  Color(0xFF22D3EE),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 20),

                // Recognition result
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(19),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        Color(0xFF29204A),
                        Color(0xFF171C35),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(23),
                    border: Border.all(
                      color: const Color(0xFF8B5CF6)
                          .withValues(alpha: 0.35),
                    ),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 53,
                        height: 53,
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [
                              Color(0xFF8B5CF6),
                              Color(0xFF6366F1),
                            ],
                          ),
                          borderRadius: BorderRadius.circular(17),
                        ),
                        child: const Icon(
                          Icons.record_voice_over_rounded,
                          color: Colors.white,
                          size: 26,
                        ),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'RECOGNIZED SIGN',
                              style: TextStyle(
                                color: Color(0xFFB7A4FF),
                                fontSize: 10,
                                fontWeight: FontWeight.w800,
                                letterSpacing: 1.2,
                              ),
                            ),
                            const SizedBox(height: 5),
                            Text(
                              _prediction,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 25,
                                fontWeight: FontWeight.w800,
                                letterSpacing: -0.4,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const Icon(
                        Icons.graphic_eq_rounded,
                        color: Color(0xFF22D3EE),
                        size: 25,
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 16),

                // Main action
                SizedBox(
                  width: double.infinity,
                  height: 57,
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: _cameraStarted
                            ? const [
                                Color(0xFFFB7185),
                                Color(0xFFEF4444),
                              ]
                            : const [
                                Color(0xFF8B5CF6),
                                Color(0xFF6366F1),
                              ],
                      ),
                      borderRadius: BorderRadius.circular(17),
                      boxShadow: [
                        BoxShadow(
                          color: (_cameraStarted
                                  ? const Color(0xFFEF4444)
                                  : const Color(0xFF8B5CF6))
                              .withValues(alpha: 0.22),
                          blurRadius: 18,
                          offset: const Offset(0, 7),
                        ),
                      ],
                    ),
                    child: ElevatedButton.icon(
                      onPressed: _cameraStarted
                          ? _stopDetection
                          : _startDetection,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.transparent,
                        foregroundColor: Colors.white,
                        shadowColor: Colors.transparent,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(17),
                        ),
                      ),
                      icon: Icon(
                        _cameraStarted
                            ? Icons.stop_circle_outlined
                            : Icons.play_circle_outline_rounded,
                        size: 23,
                      ),
                      label: Text(
                        _cameraStarted
                            ? 'STOP DETECTION'
                            : 'START SIGN DETECTION',
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 0.8,
                        ),
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 24),

                const Text(
                  'MADE FOR EVERYDAY COMMUNICATION',
                  style: TextStyle(
                    color: Color(0xFF7F8AA5),
                    fontSize: 9,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 1.25,
                  ),
                ),

                const SizedBox(height: 11),

                _infoCard(
                  icon: Icons.back_hand_rounded,
                  title: 'Gesture recognition',
                  subtitle: 'Tracks hand landmarks in real time.',
                  accent: const Color(0xFF22D3EE),
                ),

                const SizedBox(height: 10),

                _infoCard(
                  icon: Icons.phone_iphone_rounded,
                  title: 'Designed for mobile',
                  subtitle:
                      'Built to bring sign communication to your phone.',
                  accent: const Color(0xFFB9A0FF),
                ),

                const SizedBox(height: 20),

                const Center(
                  child: Text(
                    'SIGN ECHO  •  YOUR HANDS, YOUR VOICE',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Color(0xFF626D87),
                      fontSize: 9,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.0,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}