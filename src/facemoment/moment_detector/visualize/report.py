"""HTML report generator for debug sessions.

Generates a static HTML file with:
- Session summary (frames, duration, FPS)
- Trigger list with thumbnails (base64 inlined)
- Extractor performance table (avg, P95, max)
- Bottleneck analysis
- Gate open percentage
"""

import base64
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def generate_report(data: Dict[str, Any], output_path: str) -> None:
    """Generate an HTML report from debug session data.

    Args:
        data: Dict with keys:
            - "backend": str (e.g. "PATHWAY")
            - "summary": dict from PathwayMonitor.get_summary()
            - "trigger_thumbs": list of (frame_idx, np.ndarray, reason)
        output_path: Path to write the HTML file.
    """
    backend = data.get("backend", "unknown")
    summary = data.get("summary", {})
    trigger_thumbs = data.get("trigger_thumbs", [])

    html = _build_html(backend, summary, trigger_thumbs)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _build_html(
    backend: str,
    summary: Dict[str, Any],
    trigger_thumbs: List[Tuple[int, np.ndarray, str]],
) -> str:
    """Build the complete HTML string."""
    total_frames = summary.get("total_frames", 0)
    wall_time = summary.get("wall_time_sec", 0)
    eff_fps = summary.get("effective_fps", 0)
    target_fps = summary.get("target_fps", 0)
    total_triggers = summary.get("total_triggers", 0)
    gate_pct = summary.get("gate_open_pct", 0)
    ext_stats = summary.get("extractor_stats", {})
    fusion_avg = summary.get("fusion_avg_ms", 0)

    # Extractor performance rows
    ext_rows = ""
    for name, stats in ext_stats.items():
        avg = stats.get("avg_ms", 0)
        p95 = stats.get("p95_ms", 0)
        mx = stats.get("max_ms", 0)
        errs = int(stats.get("errors", 0))
        bar_pct = min(100, avg)
        color = "#4caf50" if avg < 30 else ("#ffc107" if avg < 60 else "#f44336")
        ext_rows += f"""
        <tr>
          <td>{name}</td>
          <td>{avg:.1f}</td>
          <td>{p95:.1f}</td>
          <td>{mx:.1f}</td>
          <td>{errs}</td>
          <td><div class="bar" style="width:{bar_pct}%;background:{color}"></div></td>
        </tr>"""

    if fusion_avg > 0:
        color = "#4caf50" if fusion_avg < 30 else ("#ffc107" if fusion_avg < 60 else "#f44336")
        ext_rows += f"""
        <tr>
          <td>fusion</td>
          <td>{fusion_avg:.1f}</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td><div class="bar" style="width:{min(100, fusion_avg)}%;background:{color}"></div></td>
        </tr>"""

    # Bottleneck
    bottleneck_html = ""
    if ext_stats:
        slowest = max(ext_stats, key=lambda k: ext_stats[k].get("avg_ms", 0))
        total_avg = sum(s.get("avg_ms", 0) for s in ext_stats.values()) + fusion_avg
        if total_avg > 0:
            pct = ext_stats[slowest].get("avg_ms", 0) / total_avg * 100
            bottleneck_html = f'<p class="bottleneck">Bottleneck: <strong>{slowest}</strong> ({pct:.0f}% of frame time)</p>'

    # Trigger thumbnails
    trigger_html = ""
    if trigger_thumbs:
        for idx, (frame_idx, thumb, reason) in enumerate(trigger_thumbs):
            b64 = _encode_thumbnail(thumb)
            trigger_html += f"""
            <div class="trigger-card">
              <img src="data:image/jpeg;base64,{b64}" alt="Trigger {idx+1}">
              <div class="trigger-info">
                <span class="trigger-num">#{idx+1}</span>
                <span class="trigger-reason">{reason}</span>
                <span class="trigger-frame">Frame {frame_idx}</span>
              </div>
            </div>"""
    else:
        trigger_html = '<p class="muted">No triggers fired during this session.</p>'

    # FPS ratio
    fps_ratio = eff_fps / target_fps if target_fps > 0 else 0
    fps_class = "good" if fps_ratio >= 0.9 else ("warn" if fps_ratio >= 0.7 else "bad")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FaceMoment Debug Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 24px; max-width: 900px; margin: 0 auto; }}
  h1 {{ color: #00d4ff; margin-bottom: 4px; font-size: 1.5em; }}
  h2 {{ color: #8ecae6; margin: 24px 0 12px; font-size: 1.15em; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  .subtitle {{ color: #888; font-size: 0.9em; margin-bottom: 20px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }}
  .stat-card {{ background: #16213e; border-radius: 8px; padding: 14px; }}
  .stat-card .label {{ color: #888; font-size: 0.8em; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 1.5em; font-weight: 600; margin-top: 4px; }}
  .good {{ color: #4caf50; }}
  .warn {{ color: #ffc107; }}
  .bad {{ color: #f44336; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th {{ text-align: left; padding: 8px; border-bottom: 2px solid #333; color: #8ecae6; font-size: 0.85em; }}
  td {{ padding: 8px; border-bottom: 1px solid #222; font-size: 0.9em; }}
  .bar {{ height: 10px; border-radius: 3px; min-width: 2px; }}
  .bottleneck {{ margin: 12px 0; padding: 10px 14px; background: #2d1b00; border-left: 3px solid #ffc107; border-radius: 4px; }}
  .trigger-grid {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }}
  .trigger-card {{ background: #16213e; border-radius: 8px; overflow: hidden; width: 140px; }}
  .trigger-card img {{ width: 140px; height: 140px; object-fit: cover; }}
  .trigger-info {{ padding: 8px; }}
  .trigger-num {{ color: #f44336; font-weight: 600; }}
  .trigger-reason {{ display: block; font-size: 0.8em; color: #aaa; margin-top: 2px; }}
  .trigger-frame {{ display: block; font-size: 0.75em; color: #666; }}
  .muted {{ color: #666; font-style: italic; }}
  .footer {{ margin-top: 32px; text-align: center; color: #444; font-size: 0.8em; }}
</style>
</head>
<body>
  <h1>FaceMoment Debug Report</h1>
  <p class="subtitle">Backend: {backend} | {total_frames} frames | {wall_time:.1f}s</p>

  <h2>Session Summary</h2>
  <div class="summary-grid">
    <div class="stat-card">
      <div class="label">Total Frames</div>
      <div class="value">{total_frames}</div>
    </div>
    <div class="stat-card">
      <div class="label">Duration</div>
      <div class="value">{wall_time:.1f}s</div>
    </div>
    <div class="stat-card">
      <div class="label">Effective FPS</div>
      <div class="value {fps_class}">{eff_fps:.1f}<span style="font-size:0.5em;color:#888"> / {target_fps:.0f}</span></div>
    </div>
    <div class="stat-card">
      <div class="label">Triggers</div>
      <div class="value">{total_triggers}</div>
    </div>
    <div class="stat-card">
      <div class="label">Gate Open</div>
      <div class="value">{gate_pct:.0f}%</div>
    </div>
  </div>

  <h2>Extractor Performance</h2>
  <table>
    <thead>
      <tr><th>Extractor</th><th>Avg (ms)</th><th>P95 (ms)</th><th>Max (ms)</th><th>Errors</th><th>Bar</th></tr>
    </thead>
    <tbody>
      {ext_rows}
    </tbody>
  </table>
  {bottleneck_html}

  <h2>Triggers</h2>
  <div class="trigger-grid">
    {trigger_html}
  </div>

  <div class="footer">
    Generated by FaceMoment Debug Report
  </div>
</body>
</html>"""


def _encode_thumbnail(thumb: np.ndarray) -> str:
    """Encode an OpenCV image as base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("ascii")
