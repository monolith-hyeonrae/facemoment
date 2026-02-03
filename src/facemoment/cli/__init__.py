"""Command-line interface for facemoment."""

import sys
import argparse
import logging

from facemoment.cli.utils import suppress_qt_warnings, StderrFilter

# Apply Qt warning suppression early
suppress_qt_warnings()
sys.stderr = StderrFilter(sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="FaceMoment - Portrait highlight capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  facemoment debug video.mp4                  # All extractors
  facemoment debug video.mp4 -e face          # Face only
  facemoment debug video.mp4 -e face,pose     # Face + Pose
  facemoment debug video.mp4 -e raw           # Raw video preview (no analysis)
  facemoment debug video.mp4 --no-ml          # Dummy mode (no ML)
  facemoment process video.mp4 -o ./clips     # Extract highlight clips
  facemoment process video.mp4 --distributed  # Distributed mode (venv workers)
  facemoment benchmark video.mp4              # Performance benchmark
""",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system info and available components",
        description="Display extractors, backends, triggers, and pipeline structure.",
    )
    info_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed info including device capabilities",
    )
    info_parser.add_argument(
        "--deps", action="store_true",
        help="Show extractor dependency graph",
    )

    # debug command (unified)
    debug_parser = subparsers.add_parser(
        "debug",
        help="Debug extractors with visualization",
        description="Run extractors on video with real-time visualization.",
    )
    debug_parser.add_argument("path", help="Path to video file")
    debug_parser.add_argument("--fps", type=float, default=10.0, help="Analysis FPS (default: 10)")
    debug_parser.add_argument(
        "-e", "--extractor",
        type=str,
        default="all",
        help="Extractor(s) to run: face, pose, quality, gesture, all, raw (default: all). Use 'raw' for video-only preview without analysis. Comma-separated for multiple.",
    )
    debug_parser.add_argument("--ml", action="store_true", dest="use_ml", default=None, help="Force ML backends")
    debug_parser.add_argument("--no-ml", action="store_false", dest="use_ml", help="Disable ML (dummy mode)")
    debug_parser.add_argument("--no-window", action="store_true", help="Disable interactive window")
    debug_parser.add_argument("--output", "-o", type=str, help="Save debug video to file")
    debug_parser.add_argument("--device", type=str, default="cuda:0", help="Device for ML (default: cuda:0)")
    debug_parser.add_argument("--trace", choices=["off", "minimal", "normal", "verbose"], default="off")
    debug_parser.add_argument("--trace-output", type=str, help="Output file for trace records (JSONL)")
    debug_parser.add_argument(
        "--profile", action="store_true",
        help="Show per-component timing information (detection, expression)"
    )
    # Distributed mode options for debug
    debug_parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed processing with VenvWorker (requires zmq)"
    )
    debug_parser.add_argument(
        "--venv-face", type=str, metavar="PATH",
        help="Path to venv for face extractor (enables VENV isolation)"
    )
    debug_parser.add_argument(
        "--venv-pose", type=str, metavar="PATH",
        help="Path to venv for pose extractor (enables VENV isolation)"
    )
    debug_parser.add_argument(
        "--venv-gesture", type=str, metavar="PATH",
        help="Path to venv for gesture extractor (enables VENV isolation)"
    )
    debug_parser.add_argument(
        "--config", type=str, metavar="PATH",
        help="Path to pipeline config YAML file"
    )
    debug_parser.add_argument(
        "--roi", type=str, metavar="X1,Y1,X2,Y2",
        help="Face analysis ROI in normalized coords (0-1). Default: 0.1,0.1,0.9,0.9 (center 80%%)"
    )
    debug_parser.add_argument(
        "--backend", choices=["pathway", "simple"], default=None,
        help="Execution backend: 'pathway' (Pathway streaming) or 'simple' (sequential). "
             "Default: inline (same extractors/fusion as pathway, smooth visualization)"
    )
    debug_parser.add_argument(
        "--report", type=str, metavar="PATH",
        help="Generate HTML debug report after session (e.g. --report report.html)"
    )

    # process command
    proc_parser = subparsers.add_parser("process", help="Process video and extract highlight clips")
    proc_parser.add_argument("path", help="Path to video file")
    proc_parser.add_argument("--fps", type=int, default=10, help="Analysis FPS (default: 10)")
    proc_parser.add_argument("--output-dir", "-o", type=str, default="./clips", help="Output directory")
    proc_parser.add_argument("--ml", action="store_true", dest="use_ml", default=None)
    proc_parser.add_argument("--no-ml", action="store_false", dest="use_ml")
    proc_parser.add_argument("--report", type=str, help="Save processing report to JSON file")
    proc_parser.add_argument("--cooldown", type=float, default=2.0, help="Trigger cooldown (default: 2.0s)")
    proc_parser.add_argument("--head-turn-threshold", type=float, default=30.0)
    proc_parser.add_argument("--trace", choices=["off", "minimal", "normal", "verbose"], default="off")
    proc_parser.add_argument("--trace-output", type=str)
    proc_parser.add_argument(
        "--backend", choices=["pathway", "simple"], default="pathway",
        help="Execution backend: 'pathway' (streaming, default) or 'simple' (sequential)"
    )
    # Distributed mode options
    proc_parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed processing with VenvWorker (requires zmq)"
    )
    proc_parser.add_argument(
        "--venv-face", type=str, metavar="PATH",
        help="Path to venv for face extractor (enables VENV isolation)"
    )
    proc_parser.add_argument(
        "--venv-pose", type=str, metavar="PATH",
        help="Path to venv for pose extractor (enables VENV isolation)"
    )
    proc_parser.add_argument(
        "--venv-gesture", type=str, metavar="PATH",
        help="Path to venv for gesture extractor (enables VENV isolation)"
    )
    proc_parser.add_argument(
        "--config", type=str, metavar="PATH",
        help="Path to pipeline config YAML file"
    )
    proc_parser.add_argument(
        "--roi", type=str, metavar="X1,Y1,X2,Y2",
        help="Face analysis ROI in normalized coords (0-1). Default: 0.1,0.1,0.9,0.9 (center 80%%)"
    )
    # Legacy options (hidden)
    proc_parser.add_argument("--faces", type=int, default=2, help=argparse.SUPPRESS)
    proc_parser.add_argument("--threshold", type=float, default=0.7, help=argparse.SUPPRESS)
    proc_parser.add_argument("--spike-prob", type=float, default=0.1, help=argparse.SUPPRESS)

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark extractor performance")
    bench_parser.add_argument("path", help="Path to video file")
    bench_parser.add_argument("--frames", type=int, default=100, help="Number of frames (default: 100)")
    bench_parser.add_argument("--device", type=str, default="cuda:0")
    bench_parser.add_argument("--skip-pose", action="store_true")
    bench_parser.add_argument("--expression-backend", choices=["auto", "hsemotion", "pyfeat", "none"], default="auto")

    args = parser.parse_args()

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    from facemoment.cli import commands

    if args.command == "info":
        commands.run_info(args)

    elif args.command == "debug":
        commands.run_debug(args)

    elif args.command == "process":
        commands.run_process(args)

    elif args.command == "benchmark":
        commands.run_benchmark(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
