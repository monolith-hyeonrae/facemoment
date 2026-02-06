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

    # Handle --graph flag
    graph_format = getattr(args, 'graph', None)
    if graph_format is not None:
        _print_flow_graph(graph_format)
        return

    # Handle --steps flag
    if getattr(args, 'steps', False):
        _print_processing_steps()
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


def _build_facemoment_flow_graph():
    """Build a FlowGraph representing the facemoment pipeline.

    Creates a graph with individual extractor nodes showing dependency
    relationships, rather than a single pipeline node.
    """
    from visualpath.flow import FlowGraph

    # Define all facemoment extractors with their dependencies
    extractor_defs = [
        ("quality", [], "QualityExtractor"),
        ("face_detect", [], "FaceDetectionExtractor"),
        ("face_classifier", ["face_detect"], "FaceClassifierExtractor"),
        ("expression", ["face_detect"], "ExpressionExtractor"),
        ("pose", [], "PoseExtractor"),
        ("gesture", [], "GestureExtractor"),
    ]

    # Check which extractors are actually available
    availability = {}
    availability["quality"] = True  # always available
    try:
        from facemoment.moment_detector.extractors.face_detect import FaceDetectionExtractor  # noqa: F401
        availability["face_detect"] = True
    except ImportError:
        availability["face_detect"] = False
    try:
        from facemoment.moment_detector.extractors.face_classifier import FaceClassifierExtractor  # noqa: F401
        availability["face_classifier"] = True
    except ImportError:
        availability["face_classifier"] = False
    try:
        from facemoment.moment_detector.extractors.expression import ExpressionExtractor  # noqa: F401
        availability["expression"] = True
    except ImportError:
        availability["expression"] = False
    try:
        from facemoment.moment_detector.extractors.pose import PoseExtractor  # noqa: F401
        availability["pose"] = True
    except ImportError:
        availability["pose"] = False
    try:
        from facemoment.moment_detector.extractors.gesture import GestureExtractor  # noqa: F401
        availability["gesture"] = True
    except ImportError:
        availability["gesture"] = False

    # Build graph manually to show dependency structure
    from visualpath.flow.nodes.source import SourceNode
    from visualpath.flow.node import FlowNode, FlowData

    # Lightweight stub node for graph visualization
    class ExtractorNode(FlowNode):
        """Stub node for visualization only."""
        def __init__(self, node_name):
            self._name = node_name
        @property
        def name(self):
            return self._name
        def process(self, data: FlowData) -> list:
            return [data]

    class FusionNode(FlowNode):
        """Stub node for visualization only."""
        def __init__(self, node_name="fusion"):
            self._name = node_name
        @property
        def name(self):
            return self._name
        def process(self, data: FlowData) -> list:
            return [data]

    graph = FlowGraph(entry_node="source")
    graph.add_node(SourceNode(name="source"))

    available_names = []
    for name, deps, _cls_name in extractor_defs:
        if not availability.get(name, False):
            continue
        # Skip if dependencies are not available
        if deps and not all(availability.get(d, False) for d in deps):
            continue
        graph.add_node(ExtractorNode(name))
        available_names.append(name)

    # Add fusion node
    graph.add_node(FusionNode("fusion"))

    # Add edges: source → root extractors, deps → dependents, all → fusion
    for name, deps, _cls_name in extractor_defs:
        if name not in available_names:
            continue
        if not deps:
            graph.add_edge("source", name)
        else:
            for dep in deps:
                if dep in available_names:
                    graph.add_edge(dep, name)

    # All leaf extractors → fusion
    for name in available_names:
        outgoing = graph.get_outgoing_edges(name)
        # Only connect to fusion if no other extractor depends on this one
        if not outgoing:
            graph.add_edge(name, "fusion")

    return graph, available_names


def _print_flow_graph(fmt: str = "ascii"):
    """Print pipeline FlowGraph visualization.

    Args:
        fmt: Output format - "ascii" for terminal, "dot" for Graphviz DOT.
    """
    try:
        from visualpath.flow import FlowGraph
    except ImportError:
        print("Error: visualpath.flow not available")
        print("  Install visualpath >= 0.2.0 for FlowGraph support")
        return

    graph, extractor_names = _build_facemoment_flow_graph()

    print("FaceMoment - Pipeline FlowGraph")
    print("=" * 60)
    print(f"  Extractors: {', '.join(extractor_names)}")
    print()

    if fmt == "dot":
        print(graph.to_dot("FaceMoment Pipeline"))
    else:
        print(graph.print_ascii())


def _print_processing_steps():
    """Print internal processing steps of each extractor."""
    print("FaceMoment - Processing Pipeline Steps")
    print("=" * 70)
    print()
    print("Complete data transformation pipeline from video source to trigger output.")
    print("Each step shows: name, description [backend], input → output (dependencies)")
    print()

    # Source preprocessing (input pipeline)
    print("[SourceProcessor] (Input Pipeline)")
    print("-" * 70)
    try:
        from facemoment.moment_detector.extractors.source import SourceProcessor
        for i, step in enumerate(SourceProcessor._STEPS, 1):
            opt_marker = " (optional)" if step.optional else ""
            backend_str = f" [{step.backend}]" if step.backend else ""
            deps_str = f" (depends: {step.depends_on})" if step.depends_on else ""
            print(f"  {i}. {step.name}{opt_marker}")
            print(f"     {step.description}{backend_str}")
            print(f"     {step.input_type} → {step.output_type}{deps_str}")
            print()
    except (ImportError, AttributeError):
        print("  (definition not available)")
    print()

    # Backend preprocessing (internal to ML backends)
    print("[BackendPreprocessor] (ML Backend Internal - for reference)")
    print("-" * 70)
    try:
        from facemoment.moment_detector.extractors.source import BackendPreprocessor
        for i, step in enumerate(BackendPreprocessor._STEPS, 1):
            opt_marker = " (optional)" if step.optional else ""
            backend_str = f" [{step.backend}]" if step.backend else ""
            deps_str = f" (depends: {step.depends_on})" if step.depends_on else ""
            print(f"  {i}. {step.name}{opt_marker}")
            print(f"     {step.description}{backend_str}")
            print(f"     {step.input_type} → {step.output_type}{deps_str}")
            print()
    except (ImportError, AttributeError):
        print("  (definition not available)")
    print()

    extractors_info = []

    def _get_steps(cls):
        """Get processing steps from class (supports both _STEPS and decorators)."""
        # Try class-level _STEPS first
        if hasattr(cls, '_STEPS'):
            return cls._STEPS
        # Try get_processing_steps (decorator-based)
        try:
            from facemoment.moment_detector.extractors.base import get_processing_steps
            return get_processing_steps(cls)
        except Exception:
            return []

    # Composite Face extractor (legacy)
    try:
        from facemoment.moment_detector.extractors.face import FaceExtractor
        extractors_info.append(("FaceExtractor", "face", _get_steps(FaceExtractor)))
    except (ImportError, AttributeError) as e:
        extractors_info.append(("FaceExtractor", "face", f"NOT AVAILABLE: {e}"))

    # Split Face extractors
    try:
        from facemoment.moment_detector.extractors.face_detect import FaceDetectionExtractor
        extractors_info.append(("FaceDetectionExtractor", "face_detect", _get_steps(FaceDetectionExtractor)))
    except (ImportError, AttributeError) as e:
        extractors_info.append(("FaceDetectionExtractor", "face_detect", f"NOT AVAILABLE: {e}"))

    try:
        from facemoment.moment_detector.extractors.expression import ExpressionExtractor
        extractors_info.append(("ExpressionExtractor", "expression", _get_steps(ExpressionExtractor)))
    except (ImportError, AttributeError) as e:
        extractors_info.append(("ExpressionExtractor", "expression", f"NOT AVAILABLE: {e}"))

    try:
        from facemoment.moment_detector.extractors.face_classifier import FaceClassifierExtractor
        extractors_info.append(("FaceClassifierExtractor", "face_classifier", _get_steps(FaceClassifierExtractor)))
    except (ImportError, AttributeError) as e:
        extractors_info.append(("FaceClassifierExtractor", "face_classifier", f"NOT AVAILABLE: {e}"))

    # PoseExtractor
    try:
        from facemoment.moment_detector.extractors.pose import PoseExtractor
        extractors_info.append(("PoseExtractor", "pose", _get_steps(PoseExtractor)))
    except (ImportError, AttributeError) as e:
        extractors_info.append(("PoseExtractor", "pose", f"NOT AVAILABLE: {e}"))

    # GestureExtractor
    try:
        from facemoment.moment_detector.extractors.gesture import GestureExtractor
        extractors_info.append(("GestureExtractor", "gesture", _get_steps(GestureExtractor)))
    except (ImportError, AttributeError) as e:
        extractors_info.append(("GestureExtractor", "gesture", f"NOT AVAILABLE: {e}"))

    # QualityExtractor (decorator-based)
    try:
        from facemoment.moment_detector.extractors.quality import QualityExtractor
        extractors_info.append(("QualityExtractor", "quality", _get_steps(QualityExtractor)))
    except (ImportError, AttributeError) as e:
        extractors_info.append(("QualityExtractor", "quality", f"NOT AVAILABLE: {e}"))

    # Print each extractor's steps with DAG info
    for class_name, name, steps in extractors_info:
        print(f"[{class_name}] (name: {name})")
        print("-" * 70)

        if isinstance(steps, str):
            print(f"  {steps}")
        else:
            for i, step in enumerate(steps, 1):
                opt_marker = " (optional)" if step.optional else ""
                backend_str = f" [{step.backend}]" if step.backend else ""
                deps_str = f" (depends: {step.depends_on})" if step.depends_on else ""
                print(f"  {i}. {step.name}{opt_marker}")
                print(f"     {step.description}{backend_str}")
                print(f"     {step.input_type} → {step.output_type}{deps_str}")
                print()

        print()

    # Print ASCII visualization
    print("=" * 70)
    print("[Complete Pipeline Visualization]")
    print("-" * 70)
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  SOURCE PREPROCESSING                                               │
  │  ┌──────────────┐   ┌───────────────┐   ┌────────┐   ┌───────────┐  │
  │  │ video_decode │ → │ frame_sampling│ → │ resize │ → │frame_create│  │
  │  │ [OpenCV]     │   │ [skip frames] │   │(option)│   │ [Frame]   │  │
  │  └──────────────┘   └───────────────┘   └────────┘   └─────┬─────┘  │
  └────────────────────────────────────────────────────────────┼────────┘
                                                               │
  Frame (BGR, original resolution)                             │
       ┌───────────────────────────────────────────────────────┘
       │
       ├────────────────────────────────┬──────────────────────────────────┐
       │                                │                                  │
       ▼                                ▼                                  ▼
  ┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
  │  FaceDetectionExt   │    │    QualityExtractor  │    │   PoseExtractor  │
  │  ┌───────────────┐  │    │ ┌──────────────────┐ │    │ ┌──────────────┐ │
  │  │ detect        │  │    │ │ grayscale_convert│ │    │ │ pose_estim   │ │
  │  │ (→640x640)    │  │    │ │ blur_analysis    │ │    │ │ upper_body   │ │
  │  │ tracking      │  │    │ │ brightness       │ │    │ │ hands_raised │ │
  │  │ roi_filter    │  │    │ │ contrast         │ │    │ │ wave_detect  │ │
  │  └───────┬───────┘  │    │ │ quality_gate     │ │    │ │ aggregation  │ │
  └──────────┼──────────┘    │ └──────────────────┘ │    │ └──────────────┘ │
             │               └──────────────────────┘    └──────────────────┘
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
  ┌────────────────┐  ┌───────────────────┐    ┌───────────────────────┐
  │ExpressionExt   │  │FaceClassifierExt  │    │   GestureExtractor    │
  │(depends:       │  │(depends:          │    │ ┌───────────────────┐ │
  │ face_detect)   │  │ face_detect)      │    │ │ hand_detection    │ │
  │ ┌────────────┐ │  │ ┌───────────────┐ │    │ │ finger_state      │ │
  │ │ expression │ │  │ │ track_update  │ │    │ │ gesture_classify  │ │
  │ │ aggregation│ │  │ │ noise_filter  │ │    │ │ aggregation       │ │
  │ └────────────┘ │  │ │ stability_chk │ │    │ └───────────────────┘ │
  └────────────────┘  │ │ role_classify │ │    └───────────────────────┘
                      │ │ role_assign   │ │
                      │ └───────────────┘ │
                      └───────────────────┘
             │                 │                           │
             └─────────────────┴───────────────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │HighlightFusion  │
                               │  (Trigger)      │
                               └─────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Clip Output    │
                               │  [visualbase]   │
                               └─────────────────┘
""")
