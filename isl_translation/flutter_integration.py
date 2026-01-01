"""
Flutter Integration Guide for ISL Translation
==============================================

This file contains:
1. Dart code for Flutter app
2. Python reference implementation for CTC decoding
3. Integration instructions

The Flutter app will:
1. Capture camera frames
2. Extract landmarks using MediaPipe (Flutter package)
3. Process features (normalize, compute velocity/acceleration)
4. Run ONNX model for CTC inference
5. Decode CTC output to text
"""

# =============================================================================
# FLUTTER CODE (save as lib/isl_translator.dart)
# =============================================================================

FLUTTER_CODE = '''
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

/// ISL Translation Service
/// 
/// Handles real-time sign language translation using:
/// - MediaPipe for landmark extraction
/// - ONNX Runtime for model inference
/// - CTC decoding for text generation
class ISLTranslator {
  late OrtSession _session;
  late List<String> _vocab;
  late Map<int, String> _idToToken;
  
  // Model constants
  static const int inputDim = 414;  // 46 landmarks × 3 coords × 3 features
  static const int blankId = 4;
  static const int padId = 0;
  static const int bosId = 2;
  static const int eosId = 3;
  
  // Landmark indices (46 total)
  // Pose: 4 (shoulders + elbows)
  // Left hand: 21
  // Right hand: 21
  static const List<int> poseIndices = [11, 12, 13, 14];
  
  // Frame buffer for temporal features
  final List<List<double>> _frameBuffer = [];
  static const int maxBufferSize = 500;
  
  /// Initialize the translator
  Future<void> initialize(String modelPath, String vocabPath) async {
    // Initialize ONNX Runtime
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    _session = OrtSession.fromFile(modelPath, sessionOptions);
    
    // Load vocabulary
    final vocabJson = await rootBundle.loadString(vocabPath);
    _vocab = List<String>.from(json.decode(vocabJson).keys);
    _idToToken = {};
    json.decode(vocabJson).forEach((key, value) {
      _idToToken[value as int] = key as String;
    });
  }
  
  /// Process a single frame of pose landmarks
  /// Returns null if buffer not ready, otherwise returns translation
  Future<String?> processFrame(Pose pose, List<Hand>? leftHand, List<Hand>? rightHand) async {
    // Extract landmarks (46 points × 3 coords = 138 values)
    final landmarks = _extractLandmarks(pose, leftHand, rightHand);
    
    // Add to buffer
    _frameBuffer.add(landmarks);
    
    // Keep buffer size manageable
    if (_frameBuffer.length > maxBufferSize) {
      _frameBuffer.removeAt(0);
    }
    
    // Need at least 4 frames for temporal features
    if (_frameBuffer.length < 4) {
      return null;
    }
    
    // Compute features (position + velocity + acceleration = 414 dims)
    final features = _computeFeatures();
    
    // Run inference
    final ctcOutput = await _runInference(features);
    
    // Decode CTC output
    final text = _decodeCTC(ctcOutput);
    
    return text;
  }
  
  /// Extract 46 landmarks from pose and hands
  List<double> _extractLandmarks(Pose pose, List<Hand>? leftHand, List<Hand>? rightHand) {
    final landmarks = <double>[];
    
    // Pose landmarks (4 points: shoulders + elbows)
    for (final idx in poseIndices) {
      final landmark = pose.landmarks[PoseLandmarkType.values[idx]];
      if (landmark != null) {
        landmarks.addAll([landmark.x, landmark.y, landmark.z]);
      } else {
        landmarks.addAll([0.0, 0.0, 0.0]);
      }
    }
    
    // Left hand (21 points)
    if (leftHand != null && leftHand.isNotEmpty) {
      for (final point in leftHand) {
        landmarks.addAll([point.x, point.y, point.z]);
      }
    } else {
      landmarks.addAll(List.filled(63, 0.0));  // 21 × 3
    }
    
    // Right hand (21 points)
    if (rightHand != null && rightHand.isNotEmpty) {
      for (final point in rightHand) {
        landmarks.addAll([point.x, point.y, point.z]);
      }
    } else {
      landmarks.addAll(List.filled(63, 0.0));  // 21 × 3
    }
    
    return landmarks;  // 138 values
  }
  
  /// Compute temporal features (position + velocity + acceleration)
  List<List<double>> _computeFeatures() {
    final T = _frameBuffer.length;
    final features = <List<double>>[];
    
    // Normalize landmarks
    final normalized = _normalizeLandmarks();
    
    for (var t = 0; t < T; t++) {
      final pos = normalized[t];  // 138 values
      
      // Velocity (gradient)
      final vel = <double>[];
      for (var i = 0; i < 138; i++) {
        if (t == 0) {
          vel.add(normalized[1][i] - normalized[0][i]);
        } else if (t == T - 1) {
          vel.add(normalized[T-1][i] - normalized[T-2][i]);
        } else {
          vel.add((normalized[t+1][i] - normalized[t-1][i]) / 2);
        }
      }
      
      // Acceleration (gradient of velocity)
      final acc = <double>[];
      for (var i = 0; i < 138; i++) {
        if (t == 0) {
          acc.add(0.0);
        } else if (t == T - 1) {
          acc.add(0.0);
        } else {
          final v1 = t > 0 ? (normalized[t][i] - normalized[t-1][i]) : 0.0;
          final v2 = t < T-1 ? (normalized[t+1][i] - normalized[t][i]) : 0.0;
          acc.add(v2 - v1);
        }
      }
      
      // Concatenate: pos + vel + acc = 414 values
      features.add([...pos, ...vel, ...acc]);
    }
    
    return features;
  }
  
  /// Normalize landmarks using shoulder reference
  List<List<double>> _normalizeLandmarks() {
    final normalized = <List<double>>[];
    
    for (final frame in _frameBuffer) {
      final normalizedFrame = List<double>.from(frame);
      
      // Get shoulder positions (first 2 landmarks)
      final leftShoulder = [frame[0], frame[1]];
      final rightShoulder = [frame[3], frame[4]];
      
      // Center point
      final centerX = (leftShoulder[0] + rightShoulder[0]) / 2;
      final centerY = (leftShoulder[1] + rightShoulder[1]) / 2;
      
      // Shoulder distance for scale
      final shoulderDist = math.sqrt(
        math.pow(leftShoulder[0] - rightShoulder[0], 2) +
        math.pow(leftShoulder[1] - rightShoulder[1], 2)
      );
      
      final scale = shoulderDist > 1e-6 ? shoulderDist : 1.0;
      
      // Normalize XY coordinates
      for (var i = 0; i < 46; i++) {
        final baseIdx = i * 3;
        normalizedFrame[baseIdx] = (frame[baseIdx] - centerX) / scale;
        normalizedFrame[baseIdx + 1] = (frame[baseIdx + 1] - centerY) / scale;
        // Z is already normalized by MediaPipe
      }
      
      normalized.add(normalizedFrame);
    }
    
    return normalized;
  }
  
  /// Run ONNX inference
  Future<List<List<double>>> _runInference(List<List<double>> features) async {
    // Convert to Float32List
    final T = features.length;
    final flatFeatures = Float32List(T * inputDim);
    
    for (var t = 0; t < T; t++) {
      for (var i = 0; i < inputDim; i++) {
        flatFeatures[t * inputDim + i] = features[t][i];
      }
    }
    
    // Create input tensor
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      flatFeatures,
      [1, T, inputDim],
    );
    
    // Run inference
    final outputs = await _session.runAsync(
      OrtRunOptions(),
      {'features': inputTensor},
    );
    
    // Get output
    final outputTensor = outputs?['ctc_log_probs'];
    final outputData = outputTensor?.value as List<List<List<double>>>;
    
    inputTensor.release();
    outputs?.forEach((_, v) => v.release());
    
    return outputData[0];  // Remove batch dimension
  }
  
  /// Decode CTC output using greedy decoding
  String _decodeCTC(List<List<double>> ctcOutput) {
    final tokens = <int>[];
    int? prevToken;
    
    for (final frame in ctcOutput) {
      // Get argmax
      var maxIdx = 0;
      var maxVal = frame[0];
      for (var i = 1; i < frame.length; i++) {
        if (frame[i] > maxVal) {
          maxVal = frame[i];
          maxIdx = i;
        }
      }
      
      // Skip blanks and repeated tokens
      if (maxIdx == blankId) {
        prevToken = null;
        continue;
      }
      if (maxIdx == prevToken) {
        continue;
      }
      if (maxIdx == padId || maxIdx == bosId || maxIdx == eosId) {
        continue;
      }
      
      tokens.add(maxIdx);
      prevToken = maxIdx;
    }
    
    // Decode tokens to text (for BPE, need SentencePiece decoder)
    // For now, just join token IDs - in production use SP decoder
    final text = tokens.map((id) => _idToToken[id] ?? '').join('');
    
    // Clean up SentencePiece artifacts
    return text.replaceAll('▁', ' ').trim();
  }
  
  /// Clear the frame buffer (call when user stops signing)
  void clearBuffer() {
    _frameBuffer.clear();
  }
  
  /// Dispose resources
  void dispose() {
    _session.release();
    OrtEnv.instance.release();
  }
}

/// Example usage in a Flutter widget
class ISLTranslationScreen extends StatefulWidget {
  @override
  _ISLTranslationScreenState createState() => _ISLTranslationScreenState();
}

class _ISLTranslationScreenState extends State<ISLTranslationScreen> {
  late ISLTranslator _translator;
  late PoseDetector _poseDetector;
  String _translatedText = '';
  bool _isInitialized = false;
  
  @override
  void initState() {
    super.initState();
    _initializeTranslator();
  }
  
  Future<void> _initializeTranslator() async {
    _translator = ISLTranslator();
    await _translator.initialize(
      'assets/models/isl_streaming_encoder.onnx',
      'assets/models/vocab.json',
    );
    
    _poseDetector = PoseDetector(
      options: PoseDetectorOptions(
        mode: PoseDetectionMode.stream,
        model: PoseModel.full,
      ),
    );
    
    setState(() => _isInitialized = true);
  }
  
  Future<void> _processImage(InputImage image) async {
    if (!_isInitialized) return;
    
    // Detect pose
    final poses = await _poseDetector.processImage(image);
    if (poses.isEmpty) return;
    
    // Process frame
    final text = await _translator.processFrame(
      poses.first,
      null,  // TODO: Add hand detection
      null,
    );
    
    if (text != null && text.isNotEmpty) {
      setState(() => _translatedText = text);
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('ISL Translation')),
      body: Column(
        children: [
          // Camera preview would go here
          Expanded(
            child: Center(
              child: _isInitialized
                  ? Text('Camera Feed')
                  : CircularProgressIndicator(),
            ),
          ),
          // Translation output
          Container(
            padding: EdgeInsets.all(16),
            color: Colors.black87,
            child: Text(
              _translatedText.isEmpty ? 'Start signing...' : _translatedText,
              style: TextStyle(color: Colors.white, fontSize: 24),
            ),
          ),
        ],
      ),
    );
  }
  
  @override
  void dispose() {
    _translator.dispose();
    _poseDetector.close();
    super.dispose();
  }
}
'''.strip()

# =============================================================================
# PUBSPEC.YAML DEPENDENCIES
# =============================================================================

FLUTTER_PUBSPEC = '''
# Add these dependencies to your pubspec.yaml

dependencies:
  flutter:
    sdk: flutter
  
  # ONNX Runtime for model inference
  onnxruntime: ^1.16.0
  
  # MediaPipe/ML Kit for pose and hand detection
  google_mlkit_pose_detection: ^0.6.0
  google_mlkit_hand_detection: ^0.6.0  # If available, or use custom
  
  # Camera
  camera: ^0.10.5+5
  
  # Path provider for model files
  path_provider: ^2.1.1
  
  # JSON parsing
  json_annotation: ^4.8.1

dev_dependencies:
  flutter_test:
    sdk: flutter
  build_runner: ^2.4.6
  json_serializable: ^6.7.1

# Assets configuration
flutter:
  assets:
    - assets/models/isl_streaming_encoder.onnx
    - assets/models/vocab.json
    - assets/models/model_config.json
    - assets/models/tokenizer/
'''.strip()

# =============================================================================
# INTEGRATION STEPS
# =============================================================================

INTEGRATION_GUIDE = '''
# Flutter Integration Guide for ISL Translation

## Step 1: Export the Model

After training, export the model:

```bash
cd isl_translation
python export_mobile.py \\
    --checkpoint checkpoints_v2/best_model.pt \\
    --output-dir mobile_export \\
    --tokenizer-dir tokenizer_model \\
    --quantize  # Optional: for smaller model size
```

## Step 2: Set Up Flutter Project

1. Create a new Flutter project or use existing one
2. Add dependencies to `pubspec.yaml` (see above)
3. Create `assets/models/` directory
4. Copy exported files:
   - `isl_streaming_encoder.onnx`
   - `vocab.json`
   - `model_config.json`
   - `tokenizer/` folder

## Step 3: Add Platform Support

### Android (android/app/build.gradle)
```gradle
android {
    defaultConfig {
        minSdkVersion 24  // Required for ONNX Runtime
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

### iOS (ios/Podfile)
```ruby
platform :ios, '12.0'

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '12.0'
    end
  end
end
```

## Step 4: MediaPipe Hand Detection

Google ML Kit doesn't have built-in hand detection. Options:

1. **Use TFLite Hand Landmarks Model**:
   - Download from MediaPipe models
   - Use `tflite_flutter` package
   
2. **Use MediaPipe Flutter Plugin** (if available):
   - Check pub.dev for community packages
   
3. **Extract hands from pose**:
   - Use wrist landmarks from pose detection
   - Less accurate but simpler

## Step 5: Real-time Processing Pipeline

```
Camera Frame (30 FPS)
        ↓
   MediaPipe Pose Detection (~15ms)
        ↓
   MediaPipe Hand Detection (~15ms)
        ↓
   Extract 46 Landmarks
        ↓
   Normalize + Compute Temporal Features
        ↓
   ONNX Inference (~10-20ms)
        ↓
   CTC Decoding
        ↓
   Display Translation
```

Target: 20-30 FPS processing

## Step 6: SentencePiece Integration

For proper BPE token decoding, you need SentencePiece in Dart.
Options:

1. **Port SentencePiece decoder to Dart** (recommended):
   - Only need the decode function
   - Parse the .model file
   
2. **Use FFI to call native SentencePiece**:
   - More complex setup
   - Better performance

3. **Pre-compute token-to-text mapping**:
   - Export full vocab with decoded strings
   - Larger file but simpler code

## Performance Tips

1. **Batch frames**: Process every 2-3 frames instead of every frame
2. **Use isolates**: Run inference in separate isolate
3. **Quantized model**: Use INT8 quantized model for 2-4x speedup
4. **GPU acceleration**: Enable NNAPI (Android) or CoreML (iOS) in ONNX Runtime

## Testing

1. Test with recorded videos first
2. Ensure consistent landmark extraction
3. Verify model output matches Python implementation
4. Test on target devices (performance varies significantly)
'''.strip()


# =============================================================================
# CTC Decoding Reference (Python)
# =============================================================================

def ctc_greedy_decode(log_probs, blank_id=4, pad_id=0):
    """
    Reference CTC greedy decoding implementation.
    
    Args:
        log_probs: (T, vocab_size) CTC log probabilities
        blank_id: Blank token ID
        pad_id: Pad token ID
        
    Returns:
        List of decoded token IDs
    """
    tokens = []
    prev_token = None
    
    for frame in log_probs:
        # Get argmax
        best_id = frame.argmax()
        
        # Skip blanks
        if best_id == blank_id:
            prev_token = None
            continue
        
        # Skip repeated tokens
        if best_id == prev_token:
            continue
        
        # Skip special tokens
        if best_id in {pad_id, 2, 3}:  # pad, bos, eos
            continue
        
        tokens.append(best_id)
        prev_token = best_id
    
    return tokens


# =============================================================================
# Save files
# =============================================================================

if __name__ == '__main__':
    import os
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    flutter_dir = os.path.join(output_dir, 'flutter_integration')
    os.makedirs(flutter_dir, exist_ok=True)
    
    # Save Dart code
    with open(os.path.join(flutter_dir, 'isl_translator.dart'), 'w') as f:
        f.write(FLUTTER_CODE)
    
    # Save pubspec snippet
    with open(os.path.join(flutter_dir, 'pubspec_dependencies.yaml'), 'w') as f:
        f.write(FLUTTER_PUBSPEC)
    
    # Save integration guide
    with open(os.path.join(flutter_dir, 'INTEGRATION_GUIDE.md'), 'w') as f:
        f.write(INTEGRATION_GUIDE)
    
    print("Flutter integration files saved to:", flutter_dir)
    print("\nFiles created:")
    print("  - isl_translator.dart (main Dart code)")
    print("  - pubspec_dependencies.yaml (Flutter dependencies)")
    print("  - INTEGRATION_GUIDE.md (setup instructions)")
