"""Info command for facemoment CLI.

Shows available extractors, backends, and pipeline structure.
"""

import sys


def run_info(args):
    """Show system information and available components."""
    # Handle --deps flag
    if getattr(args, 'deps', False):
        _print_dependency_graph()
        return

    print("FaceMoment - System Information")
    print("=" * 60)

    # Version info
    _print_version_info()

    # Extractor availability
    print("\n[Extractors]")
    print("-" * 60)
    _check_face_extractor(verbose=args.verbose)
    _check_pose_extractor(verbose=args.verbose)
    _check_gesture_extractor(verbose=args.verbose)
    _check_quality_extractor()

    # Fusion info
    print("\n[Fusion]")
    print("-" * 60)
    _print_fusion_info()

    # Trigger types
    print("\n[Trigger Types]")
    print("-" * 60)
    _print_trigger_types()

    # Pipeline structure
    print("\n[Pipeline Structure]")
    print("-" * 60)
    _print_pipeline_structure()

    # Device info
    if args.verbose:
        print("\n[Device]")
        print("-" * 60)
        _print_device_info()


def _print_version_info():
    """Print version information."""
    try:
        from facemoment import __version__
        print(f"  facemoment: {__version__}")
    except ImportError:
        print("  facemoment: (version not available)")

    try:
        import visualbase
        version = getattr(visualbase, '__version__', 'installed')
        print(f"  visualbase: {version}")
    except ImportError:
        print("  visualbase: NOT INSTALLED")


def _check_face_extractor(verbose: bool = False):
    """Check FaceExtractor availability."""
    status = {"detection": None, "expression": None}

    # Detection backend (InsightFace)
    try:
        import insightface
        from facemoment.moment_detector.extractors.backends.face_backends import InsightFaceSCRFD
        status["detection"] = f"InsightFace SCRFD (v{insightface.__version__})"
    except ImportError as e:
        status["detection"] = f"NOT AVAILABLE (insightface not installed)"
    except Exception as e:
        status["detection"] = f"ERROR: {e}"

    # Expression backend (HSEmotion or PyFeat)
    try:
        from facemoment.moment_detector.extractors.backends.face_backends import HSEmotionBackend
        import hsemotion_onnx
        status["expression"] = "HSEmotion (fast)"
    except ImportError:
        try:
            from facemoment.moment_detector.extractors.backends.face_backends import PyFeatBackend
            status["expression"] = "PyFeat (accurate, slow)"
        except ImportError:
            status["expression"] = "NOT AVAILABLE (install hsemotion-onnx or py-feat)"

    available = status["detection"] and "NOT" not in status["detection"]
    icon = "+" if available else "-"
    print(f"  [{icon}] FaceExtractor")
    print(f"        Detection:  {status['detection']}")
    print(f"        Expression: {status['expression']}")


def _check_pose_extractor(verbose: bool = False):
    """Check PoseExtractor availability."""
    try:
        import ultralytics
        from facemoment.moment_detector.extractors.backends.pose_backends import YOLOPoseBackend
        print(f"  [+] PoseExtractor")
        print(f"        Backend: YOLO-Pose (ultralytics v{ultralytics.__version__})")
    except ImportError:
        print(f"  [-] PoseExtractor")
        print(f"        Backend: NOT AVAILABLE (ultralytics not installed)")


def _check_gesture_extractor(verbose: bool = False):
    """Check GestureExtractor availability."""
    try:
        import mediapipe
        print(f"  [+] GestureExtractor")
        print(f"        Backend: MediaPipe Hands (v{mediapipe.__version__})")
    except ImportError:
        print(f"  [-] GestureExtractor")
        print(f"        Backend: NOT AVAILABLE (mediapipe not installed)")


def _check_quality_extractor():
    """Check QualityExtractor (always available)."""
    print(f"  [+] QualityExtractor")
    print(f"        Backend: OpenCV (blur/brightness/contrast)")


def _print_fusion_info():
    """Print fusion module information."""
    print("  HighlightFusion")
    print("    - Gate: quality + face conditions (hysteresis)")
    print("    - EWMA smoothing for stable signals")
    print("    - Configurable cooldown between triggers")


def _print_trigger_types():
    """Print available trigger types."""
    triggers = [
        ("expression_spike", "FaceExtractor", "Sudden expression change"),
        ("head_turn", "FaceExtractor", "Fast head rotation"),
        ("hand_wave", "PoseExtractor", "Hand waving motion"),
        ("camera_gaze", "FaceExtractor", "Looking at camera"),
        ("passenger_interaction", "FaceExtractor", "Passengers facing each other"),
        ("gesture_vsign", "GestureExtractor", "V-sign gesture"),
        ("gesture_thumbsup", "GestureExtractor", "Thumbs up gesture"),
    ]

    for trigger, source, desc in triggers:
        print(f"  {trigger:24s} [{source:16s}] {desc}")


def _print_pipeline_structure():
    """Print pipeline structure diagram."""
    print("""
  Video Source (visualbase)
       │
       ▼
  ┌─────────────────────────────────────────┐
  │              Extractors                 │
  │  ┌─────────┐ ┌─────────┐ ┌───────────┐  │
  │  │  Face   │ │  Pose   │ │  Quality  │  │
  │  │(detect+ │ │ (YOLO)  │ │(blur/bright)│ │
  │  │express) │ │         │ │           │  │
  │  └────┬────┘ └────┬────┘ └─────┬─────┘  │
  │       │           │            │        │
  │       └───────────┴────────────┘        │
  │                   │                     │
  │                   ▼                     │
  │  ┌─────────────────────────────────┐    │
  │  │       HighlightFusion           │    │
  │  │  Gate Check → Signal Analysis   │    │
  │  │  → Trigger Decision             │    │
  │  └─────────────┬───────────────────┘    │
  └────────────────┼────────────────────────┘
                   │
                   ▼
             Trigger Event
                   │
                   ▼
            Clip Extraction
""")


def _print_device_info():
    """Print device information."""
    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  cuda:{i} - {name} ({mem:.1f} GB)")
        else:
            print("  CUDA: Not available")
    except ImportError:
        print("  PyTorch: Not installed")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  ONNX Runtime: {', '.join(providers)}")
    except ImportError:
        print("  ONNX Runtime: Not installed")


def _print_dependency_graph():
    """Print extractor dependency graph."""
    print("FaceMoment - Extractor Dependency Graph")
    print("=" * 60)

    # Define extractors with their dependencies
    extractors = [
        ("face_detect", [], "Face detection (bbox, head pose)"),
        ("face_classifier", ["face_detect"], "Face role classification (main/passenger/transient)"),
        ("expression", ["face_detect"], "Expression analysis (emotions)"),
        ("face", [], "Face composite (detect + expression)"),
        ("pose", [], "Pose estimation (keypoints)"),
        ("gesture", [], "Gesture detection (hand signs)"),
        ("quality", [], "Quality metrics (blur, brightness)"),
        ("dummy", [], "Dummy extractor (testing)"),
    ]

    # Build dependency graph
    print("\n[Dependency Tree]")
    print("-" * 60)

    # Group by root (no dependencies)
    roots = [(name, deps, desc) for name, deps, desc in extractors if not deps]
    dependents = [(name, deps, desc) for name, deps, desc in extractors if deps]

    for name, deps, desc in roots:
        print(f"  {name}")
        print(f"  │   {desc}")
        # Find children
        children = [(n, d, ds) for n, d, ds in dependents if name in d]
        for i, (child_name, child_deps, child_desc) in enumerate(children):
            is_last = i == len(children) - 1
            prefix = "└──" if is_last else "├──"
            print(f"  {prefix} {child_name}")
            print(f"  {'    ' if is_last else '│   '}   {child_desc}")
        print()

    # Execution order recommendation
    print("[Recommended Execution Order]")
    print("-" * 60)
    print("  For face analysis pipeline:")
    print("    1. face_detect     → 2. face_classifier → 3. expression → 4. fusion")
    print()
    print("  For full analysis pipeline:")
    print("    1. quality         (gate check)")
    print("    2. face_detect     (detection)")
    print("    3. face_classifier (depends on face_detect)")
    print("    4. expression      (depends on face_detect)")
    print("    5. pose            (independent)")
    print("    6. gesture         (independent)")
    print("    7. fusion          (combines all)")
    print()

    # Visualize as ASCII graph
    print("[ASCII Graph]")
    print("-" * 60)
    print("""
  ┌─────────────┐
  │   quality   │ (no deps)
  └──────┬──────┘
         │
  ┌──────┴──────┐
  │ face_detect │ (no deps)
  └──────┬──────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  ┌──────────────┐  ┌─────────────┐
  │  classifier  │  │ expression  │  depends: [face_detect]
  │ (main/pass)  │  │ (emotions)  │
  └──────┬───────┘  └──────┬──────┘
         │                 │
         └────────┬────────┘
                  │
  ┌───────────────┼───────────────┐
  │               │               │
  ▼               ▼               ▼
┌──────┐     ┌─────────┐     ┌─────────┐
│ pose │     │ gesture │     │ fusion  │
└──────┘     └─────────┘     └─────────┘
""")
