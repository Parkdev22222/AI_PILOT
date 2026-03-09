#!/usr/bin/env python
import argparse
import json
import os
import sys
import threading
import socket
from functools import partial
from html import escape
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from command.commander_db import CommanderCombatDB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)

    # Gradio web server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)

    # Offline map assets
    # Example:
    #   /data/offline_map/leaflet/leaflet.js, leaflet.css, images/...
    #   /data/offline_map/tiles/{z}/{x}/{y}.png
    parser.add_argument("--assets-root", type=str, default="/data/offline_map")
    parser.add_argument("--static-port", type=int, default=8090)

    # What IP should the CLIENT browser use to access static server?
    # If empty, we'll try: env HOST_IP -> auto-detect -> fallback 127.0.0.1
    parser.add_argument("--host-ip", type=str, default="")
    return parser.parse_args()


def _pick_host_ip(args_host: str, override: str = "") -> str:
    """
    Choose a browser-reachable IP for tile/leaflet static server.
    - If override provided, use it.
    - Else if env HOST_IP exists, use it.
    - Else if args_host is a concrete IP/hostname (not 0.0.0.0), use it.
    - Else auto-detect local LAN IP.
    """
    if override:
        return override.strip()
    env_ip = os.environ.get("HOST_IP", "").strip()
    if env_ip:
        return env_ip

    if args_host and args_host != "0.0.0.0":
        return args_host

    # Auto detect outbound interface IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _start_static_server(root_dir: str, port: int):
    """
    Serve offline assets via HTTP:
      http://<HOST_IP>:<port>/tiles/{z}/{x}/{y}.png
      http://<HOST_IP>:<port>/leaflet/leaflet.js
      http://<HOST_IP>:<port>/leaflet/leaflet.css
      http://<HOST_IP>:<port>/leaflet/images/...
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"--assets-root not found: {root_dir}\n"
            "It must contain:\n"
            "  leaflet/leaflet.js, leaflet.css, images/...\n"
            "  tiles/{z}/{x}/{y}.png\n"
        )

    handler = partial(SimpleHTTPRequestHandler, directory=root_dir)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)

    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd


def _format_events(state):
    events = state.get("events", [])
    filtered = []
    for ev in events:
        et = ev.get("event_type", "")
        if et in {"ENGAGEMENT_START", "RTB_DECISION", "ALLY_SHOTDOWN", "ENEMY_SHOTDOWN"}:
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
                    f"- [step {step}] LLM 복귀명령 | ally={escape(str(payload.get('ally', '-')))} "
                    f"decision={escape(str(payload.get('decision', '-')))}"
                )
    return "\n".join(lines)


def _leaflet_map_html(tile_base_url: str, leaflet_base_url: str):
    """
    Create an offline Leaflet map that loads local tiles and supports zoom/pan/scroll.
    The map is initialized only once, and updates happen via window.__AC_UPDATE_MAP(stateJsonStr).
    """
    return f"""
<div id="ac-map" style="height:78vh;border:1px solid #ddd;border-radius:8px;position:relative;overflow:hidden;"></div>

<link rel="stylesheet" href="{leaflet_base_url}/leaflet.css">
<script src="{leaflet_base_url}/leaflet.js"></script>

<script>
(function() {{
  if (window.__AC_MAP_INITIALIZED) return;
  window.__AC_MAP_INITIALIZED = true;

  const el = document.getElementById("ac-map");
  if (!el) return;

  // Init
  const map = L.map("ac-map", {{
    zoomControl: true,
    attributionControl: false,
    preferCanvas: true
  }}).setView([36.3, 127.8], 7);

  // Local tiles
  L.tileLayer("{tile_base_url}", {{
    minZoom: 3,
    maxZoom: 14,
    tileSize: 256,
    updateWhenIdle: true,
    keepBuffer: 6
  }}).addTo(map);

  // Layers for dynamic overlays
  const gEng = L.layerGroup().addTo(map);
  const gAlly = L.layerGroup().addTo(map);
  const gEnemy = L.layerGroup().addTo(map);

  // Update function called from Gradio JS hook
  window.__AC_UPDATE_MAP = function(stateJsonStr) {{
    try {{
      const s = JSON.parse(stateJsonStr || "{{}}");
      const tracks = Array.isArray(s.tracks) ? s.tracks : [];
      const engagements = Array.isArray(s.engagements) ? s.engagements : [];

      gEng.clearLayers();
      gAlly.clearLayers();
      gEnemy.clearLayers();

      // Engagement circles
      for (const e of engagements) {{
        const lon = e.center_lon, lat = e.center_lat;
        if (typeof lon !== "number" || typeof lat !== "number") continue;

        const label = "교전중 " + (e.region || "-") + " (" + (e.env_id || "-") + ")";

        L.circle([lat, lon], {{
          radius: 20000, // 20km (meters)
          color: "#c40000",
          weight: 2,
          fillColor: "#ff0000",
          fillOpacity: 0.20
        }}).bindTooltip(label, {{sticky:true}}).addTo(gEng);
      }}

      function center(list) {{
        if (!list.length) return null;
        let slon=0, slat=0;
        for (const t of list) {{ slon += t.lon; slat += t.lat; }}
        return [slat/list.length, slon/list.length]; // [lat, lon]
      }}

      const transit = tracks.filter(t =>
        t && t.status === "TRANSIT" &&
        typeof t.lon === "number" && typeof t.lat === "number"
      );

      const groups = {{}};
      for (const t of transit) {{
        const gid = (t.group_id ?? "").toString();
        if (!groups[gid]) groups[gid] = [];
        groups[gid].push(t);
      }}

      for (const gid in groups) {{
        const g = groups[gid];
        const allies = g.filter(t => t.team === "ally");
        const enemies = g.filter(t => t.team === "enemy");

        const ac = center(allies);
        const ec = center(enemies);

        if (ac) {{
          L.circleMarker(ac, {{
            radius: 10,
            color: "#1f77ff",
            weight: 2,
            fillColor: "#1f77ff",
            fillOpacity: 0.35
          }}).bindTooltip(`아군 편대(${{gid}})`, {{sticky:true}}).addTo(gAlly);
        }}

        if (ec) {{
          L.circleMarker(ec, {{
            radius: 10,
            color: "#111111",
            weight: 2,
            fillColor: "#111111",
            fillOpacity: 0.35
          }}).bindTooltip(`적군 편대(${{gid}})`, {{sticky:true}}).addTo(gEnemy);
        }}
      }}
    }} catch (err) {{
      console.warn("Map update failed:", err);
    }}
  }};
}})();
</script>
"""


def build_dashboard(db_path: str, run_id: str, args):
    import gradio as gr

    db = CommanderCombatDB(db_path)

    # 1) Start local static server for leaflet + tiles
    _start_static_server(args.assets_root, args.static_port)

    # 2) Build URLs that the CLIENT browser can reach
    host_ip = _pick_host_ip(args.host, args.host_ip)
    tile_url = f"http://{host_ip}:{args.static_port}/tiles/{{z}}/{{x}}/{{y}}.png"
    leaflet_url = f"http://{host_ip}:{args.static_port}/leaflet"

    # 3) Send minimal JSON (tracks + engagements) to browser; browser updates map without re-init
    def refresh_state_json():
        state = db.get_live_state(run_id)
        status = (
            f"run={state.get('run_id','-')} | step={state.get('global_step',0)} | "
            f"transit_tracks={len([t for t in state.get('tracks',[]) if t.get('status')=='TRANSIT'])} | "
            f"engagements={len(state.get('engagements', []))}"
        )

        payload = {
            "tracks": state.get("tracks") or [],
            "engagements": state.get("engagements") or [],
        }
        return json.dumps(payload, ensure_ascii=False), _format_events(state), status

    # JS hook: called after Python refresh; updates existing map via global function
    JS_UPDATE = """
    (state_json) => {
      try {
        if (window.__AC_UPDATE_MAP) window.__AC_UPDATE_MAP(state_json);
      } catch (e) {
        console.warn(e);
      }
      return [];
    }
    """

    with gr.Blocks(title="Air Commander Live Dashboard") as demo:
        gr.Markdown("## Air Commander Live Dashboard (Gradio)")

        # Map: load ONCE
        map_value = _leaflet_map_html(tile_url, leaflet_url)
        try:
            map_html = gr.HTML(map_value, sanitize=False)  # gradio 4.x
        except TypeError:
            map_html = gr.HTML(map_value)  # older gradio (no sanitize param)

        # Hidden state carrier for JS update
        state_json = gr.Textbox(visible=False)

        with gr.Row():
            with gr.Column(scale=3):
                map_html
            with gr.Column(scale=2):
                event_md = gr.Markdown(label="이벤트")

        status_tb = gr.Textbox(label="상황 요약", interactive=False)

        # Auto-refresh every 1s with compatibility fallbacks
        def _bind_refresh_load():
            # 1) Try demo.load(..., every=1.0, js=...)
            try:
                demo.load(
                    refresh_state_json,
                    outputs=[state_json, event_md, status_tb],
                    every=1.0,
                    js=JS_UPDATE,
                )
                return
            except TypeError:
                pass

            # 2) Try Timer / Interval
            if hasattr(gr, "Timer"):
                t = gr.Timer(1.0)
                try:
                    t.tick(refresh_state_json, outputs=[state_json, event_md, status_tb], js=JS_UPDATE)
                except TypeError:
                    # very old versions may not support js=
                    t.tick(refresh_state_json, outputs=[state_json, event_md, status_tb])
                return

            if hasattr(gr, "Interval"):
                itv = gr.Interval(1.0)
                try:
                    itv.tick(refresh_state_json, outputs=[state_json, event_md, status_tb], js=JS_UPDATE)
                except TypeError:
                    itv.tick(refresh_state_json, outputs=[state_json, event_md, status_tb])
                return

            # 3) Manual fallback
            gr.Markdown("⚠️ 이 Gradio 버전은 자동 갱신이 제한되어 수동 갱신만 지원됩니다.")
            btn = gr.Button("새로고침")
            try:
                btn.click(refresh_state_json, outputs=[state_json, event_md, status_tb], js=JS_UPDATE)
            except TypeError:
                btn.click(refresh_state_json, outputs=[state_json, event_md, status_tb])

        _bind_refresh_load()

    return demo


def main():
    args = parse_args()
    try:
        demo = build_dashboard(args.db_path, args.run_id, args)
    except ModuleNotFoundError as exc:
        raise RuntimeError("gradio is required for run_commander_web.py. Please install gradio in this environment.") from exc

    demo.launch(server_name=args.host, server_port=args.port, show_api=False)


if __name__ == "__main__":
    main()
