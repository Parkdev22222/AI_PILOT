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
<div id="map" style="height:78vh;border:1px solid #ddd;border-radius:8px;"></div>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
(function() {{
  const data = {payload};
  const mapEl = document.getElementById('map');
  mapEl.innerHTML = '';
  const map = L.map('map').setView([36.2, 127.8], 7);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 12 }}).addTo(map);

  const engagementLayer = L.layerGroup().addTo(map);
  const transitLayer = L.layerGroup().addTo(map);

  for (const e of (data.engagements || [])) {{
    L.circle([e.center_lat, e.center_lon], {{
      radius: 20000,
      color: '#ff0000',
      fillColor: '#ff0000',
      fillOpacity: 0.25,
      weight: 1,
    }}).bindTooltip(`교전중 ${{e.region}} (${{e.env_id}})`).addTo(engagementLayer);
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
      L.circleMarker([ac.lat, ac.lon], {{
        radius: 12, color: '#1f77ff', fillColor: '#1f77ff', fillOpacity: 0.35, weight: 2,
      }}).bindTooltip(`아군 편대(${{gid}})`).addTo(transitLayer);
    }}
    if (ec) {{
      L.circleMarker([ec.lat, ec.lon], {{
        radius: 12, color: '#111111', fillColor: '#111111', fillOpacity: 0.35, weight: 2,
      }}).bindTooltip(`적군 편대(${{gid}})`).addTo(transitLayer);
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
