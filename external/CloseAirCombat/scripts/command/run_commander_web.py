#!/usr/bin/env python
import argparse
import json
import os
import sys
from html import escape

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from command.commander_db import CommanderCombatDB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    return parser.parse_args()


def _build_map_html(state):
    tracks = state.get("tracks", [])
    engagements = state.get("engagements", [])
    payload = json.dumps({"tracks": tracks, "engagements": engagements}, ensure_ascii=False)

    return f"""
<div id="map" style="height:78vh;border:1px solid #ddd;border-radius:8px;position:relative;background:#f6f8fb;"></div>
<script>
(function() {{
  const data = {payload};
  const mapEl = document.getElementById('map');
  mapEl.innerHTML = '';

  const canvas = document.createElement('canvas');
  canvas.width = mapEl.clientWidth || 900;
  canvas.height = mapEl.clientHeight || 600;
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  mapEl.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // Offline-friendly Korea bounding box projection (no external map/tile dependency).
  const BOUNDS = {{ minLon: 124.5, maxLon: 132.5, minLat: 33.0, maxLat: 39.8 }};
  function project(lon, lat) {{
    const x = ((lon - BOUNDS.minLon) / (BOUNDS.maxLon - BOUNDS.minLon)) * canvas.width;
    const y = canvas.height - ((lat - BOUNDS.minLat) / (BOUNDS.maxLat - BOUNDS.minLat)) * canvas.height;
    return [x, y];
  }}
  function kmToPixels(km) {{
    const lonSpanKm = (BOUNDS.maxLon - BOUNDS.minLon) * 88.0; // rough conversion near Korea latitude.
    return (km / lonSpanKm) * canvas.width;
  }}

  function drawBaseMap() {{
    ctx.fillStyle = '#eef3f9';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // grid
    ctx.strokeStyle = '#d6deea';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 8; i++) {{
      const x = (i / 8) * canvas.width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }}
    for (let i = 0; i <= 6; i++) {{
      const y = (i / 6) * canvas.height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }}

    // Simplified peninsula polyline for operator orientation.
    const koreaOutline = [
      [126.0, 34.2], [126.8, 35.1], [127.7, 36.0], [128.8, 37.0],
      [129.8, 38.2], [128.9, 39.0], [127.0, 38.6], [126.0, 37.7],
      [125.4, 36.3], [125.6, 35.0], [126.0, 34.2],
    ];
    ctx.beginPath();
    koreaOutline.forEach((p, idx) => {{
      const [x, y] = project(p[0], p[1]);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }});
    ctx.closePath();
    ctx.fillStyle = '#dfe8d8';
    ctx.strokeStyle = '#9fb090';
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  }}

  function drawLabel(text, x, y) {{
    ctx.font = '12px sans-serif';
    const pad = 3;
    const w = ctx.measureText(text).width + 2 * pad;
    const h = 16;
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.fillRect(x + 6, y - h / 2, w, h);
    ctx.fillStyle = '#222';
    ctx.fillText(text, x + 6 + pad, y + 4);
  }}

  drawBaseMap();

  for (const e of (data.engagements || [])) {{
    const [x, y] = project(e.center_lon, e.center_lat);
    const r = kmToPixels(20);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,0,0,0.25)';
    ctx.strokeStyle = 'rgba(200,0,0,0.8)';
    ctx.lineWidth = 1.5;
    ctx.fill();
    ctx.stroke();
    drawLabel(`교전중 ${{e.region}} (${{e.env_id}})`, x, y);
  }}

  function groupBy(arr, key) {{
    return arr.reduce((acc, x) => {{
      const k = x[key] || '';
      if (!acc[k]) acc[k] = [];
      acc[k].push(x);
      return acc;
    }}, {{}});
  }}
  function center(tracks) {{
    if (!tracks.length) return null;
    const lon = tracks.reduce((s, t) => s + t.lon, 0) / tracks.length;
    const lat = tracks.reduce((s, t) => s + t.lat, 0) / tracks.length;
    return {{ lon, lat }};
  }}

  const transit = (data.tracks || []).filter(t => t.status === 'TRANSIT');
  const grouped = groupBy(transit, 'group_id');
  for (const [gid, tracks] of Object.entries(grouped)) {{
    const allies = tracks.filter(t => t.team === 'ally');
    const enemies = tracks.filter(t => t.team === 'enemy');
    const ac = center(allies);
    const ec = center(enemies);

    if (ac) {{
      const [x, y] = project(ac.lon, ac.lat);
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(31,119,255,0.35)';
      ctx.strokeStyle = '#1f77ff';
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
      drawLabel(`아군 편대(${{gid}})`, x, y);
    }}
    if (ec) {{
      const [x, y] = project(ec.lon, ec.lat);
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(17,17,17,0.35)';
      ctx.strokeStyle = '#111111';
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
      drawLabel(`적군 편대(${{gid}})`, x, y);
    }}
  }}
}})();
</script>
"""


def _format_events(state):
    events = state.get("events", [])
    filtered = []
    for ev in events:
        et = ev.get("event_type", "")
        if et in {"ENGAGEMENT_START", "RTB_DECISION", "ALLY_SHOTDOWN", "ENEMY_SHOTDOWN"}:
            filtered.append(ev)
        elif et == "RTB_DECISION":
            filtered.append(ev)

    # keep latest first, max 20
    filtered = sorted(filtered, key=lambda x: int(x.get("step", 0)), reverse=True)[:20]

    lines = ["### 이벤트 발생 여부", "- 교전 발생", "- 아군 격추", "- 적군 격추", "- LLM 아군 복귀 명령", "", "---", ""]
    if not filtered:
        lines.append("최근 이벤트 없음")
    else:
        for ev in filtered:
            et = ev.get("event_type", "")
            step = ev.get("step", "-")
            region = ev.get("region") or "-"
            payload = ev.get("payload", {})
            if et == "ENGAGEMENT_START":
                lines.append(f"- [step {step}] 교전 발생 | region={escape(str(region))}")
            elif et == "ALLY_SHOTDOWN":
                lines.append(f"- [step {step}] 아군 격추 | unit={escape(str(payload.get('unit_id', '-')))}")
            elif et == "ENEMY_SHOTDOWN":
                lines.append(f"- [step {step}] 적군 격추 | unit={escape(str(payload.get('unit_id', '-')))}")
            elif et == "RTB_DECISION":
                lines.append(
                    f"- [step {step}] LLM 복귀명령 | ally={escape(str(payload.get('ally', '-')))} decision={escape(str(payload.get('decision', '-')))}"
                )
    return "\n".join(lines)


def build_dashboard(db_path: str, run_id: str):
    import gradio as gr
    db = CommanderCombatDB(db_path)

    def refresh():
        state = db.get_live_state(run_id)
        status = (
            f"run={state.get('run_id','-')} | step={state.get('global_step',0)} | "
            f"transit_tracks={len([t for t in state.get('tracks',[]) if t.get('status')=='TRANSIT'])} | "
            f"engagements={len(state.get('engagements', []))}"
        )
        return _build_map_html(state), _format_events(state), status

    with gr.Blocks(title="Air Commander Live Dashboard") as demo:
        gr.Markdown("## Air Commander Live Dashboard (Gradio)")
        with gr.Row():
            with gr.Column(scale=3):
                map_html = gr.HTML(label="한반도 실시간 지도")
            with gr.Column(scale=2):
                event_md = gr.Markdown(label="이벤트")
        status_tb = gr.Textbox(label="상황 요약", interactive=False)

        demo.load(refresh, outputs=[map_html, event_md, status_tb], every=1.0)

    return demo


def main():
    args = parse_args()
    try:
        demo = build_dashboard(args.db_path, args.run_id)
    except ModuleNotFoundError as exc:
        raise RuntimeError("gradio is required for run_commander_web.py. Please install gradio in this environment.") from exc
    demo.launch(server_name=args.host, server_port=args.port, show_api=False)


if __name__ == "__main__":
    main()
