"""
gradio_dashboard.py
===================
전술 공중전 시뮬레이터 Gradio 대시보드

지도
  - 아군(파란색) / 적군(빨간색) 기체 실시간 위치 (Plotly Scattergeo, 완전 오프라인)
  - 비행 단계별 마커 심볼: 접근·교전=▲, RTB=▽, 재장착=●, 임무완료=✕
  - 교전 구역 → 빨간 반투명 20km 원
  - 이벤트 로그 테이블

전투 현황
  - 생존율 막대 (아군/적군)
  - LLM(EXAONE-3.5) 판단 이력

DB 폴링 방식: gr.Timer(every=N) — 완전 로컬 SQLite, 외부 네트워크 불필요
"""

import base64
import io
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .combat_db import CombatDB
from .llm_commander import ENEMY_BASES, FRIENDLY_BASES

# ── 기지 정보 (고정 마커) ────────────────────────────────────────────────────

_ALL_BASES = {
    **{k: {**v, "team": "enemy"}   for k, v in ENEMY_BASES.items()},
    **{k: {**v, "team": "friendly"} for k, v in FRIENDLY_BASES.items()},
}

# 비행 단계 → matplotlib 마커
_PHASE_MARKER = {
    "approach":  "^",
    "combat":    "^",
    "rtb_loss":  "v",
    "reload":    "o",
    "returning": "^",
    "support":   "^",
    "done":      "x",
    "unknown":   "o",
}

# 비행 단계 한글 이름
_PHASE_KO = {
    "approach":  "접근",
    "combat":    "교전",
    "rtb_loss":  "RTB(손실)",
    "reload":    "재장착",
    "returning": "복귀중",
    "support":   "지원",
    "done":      "임무완료",
    "unknown":   "-",
}

_EVENT_TYPE_KO = {
    "major_loss":               "전력 50% 손실",
    "ammo_depleted":            "무장 고갈",
    "enemy_formation_destroyed": "적 편대 전멸",
}


# ─────────────────────────────────────────────────────────────────────────────
# 지리 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0, a)))


def _circle_coords(
    center_lon: float, center_lat: float,
    radius_km: float, n: int = 72,
) -> Tuple[List[float], List[float]]:
    """위경도 원 좌표 계산 (소원 근사)."""
    theta = np.linspace(0, 2 * math.pi, n + 1)
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(center_lat)) + 1e-9)
    lons = (center_lon + dlon * np.cos(theta)).tolist()
    lats = (center_lat + dlat * np.sin(theta)).tolist()
    return lons, lats


# ─────────────────────────────────────────────────────────────────────────────
# 대시보드 클래스
# ─────────────────────────────────────────────────────────────────────────────

class TacticalDashboard:
    """
    Parameters
    ----------
    db_path           : SQLite DB 파일 경로
    sim_id            : 시뮬레이션 ID (TacticalController.sim_id)
    refresh_interval  : 자동 갱신 주기 (초)
    map_center        : (lon, lat) 지도 초기 중심
    map_zoom          : 지도 초기 줌
    """

    def __init__(
        self,
        db_path: str = "combat_simulation.db",
        sim_id: int = 1,
        refresh_interval: float = 2.0,
        start_callback=None,
    ):
        self.db = CombatDB(db_path)
        self.sim_id = sim_id
        self.refresh_interval = refresh_interval
        self.start_callback = start_callback
        self._sim_started = False

    # ------------------------------------------------------------------
    # 데이터 조회
    # ------------------------------------------------------------------

    def _states(self) -> List[Dict]:
        try:
            return self.db.get_latest_aircraft_states(self.sim_id)
        except Exception:
            return []

    def _formations(self) -> List[Dict]:
        try:
            return self.db.get_formations(self.sim_id)
        except Exception:
            return []

    def _events(self) -> List[Dict]:
        try:
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM events WHERE sim_id=? ORDER BY step ASC LIMIT 100",
                (self.sim_id,),
            ).fetchall()
            conn.close()
            result = [dict(r) for r in rows]
            for r in result:
                try:
                    r["details_json"] = json.loads(r["details_json"])
                except Exception:
                    pass
                try:
                    r["llm_decision"] = json.loads(r["llm_decision"]) if r["llm_decision"] else None
                except Exception:
                    pass
            return result
        except Exception:
            return []

    def _llm_decisions(self) -> List[Dict]:
        try:
            return self.db.get_recent_decisions(self.sim_id, limit=50)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 교전 구역 감지
    # ------------------------------------------------------------------

    def _detect_combat_zones(self, states: List[Dict]) -> List[Dict]:
        """
        LLM이 assign_formation_targets()로 짝지은 아군-적군 편대 쌍 기준으로
        교전 구역을 계산. DB formation_info.paired_enemy_id 를 사용.
        [{center_lon, center_lat, radius_km, label}, …]
        """
        if not states:
            return []

        # ── LLM 배정 기반: formation_info 의 paired_enemy_id 활용 ────────
        formations = self._formations()
        friendly_fmts = [f for f in formations if f["team"] == "friendly"
                         and f.get("paired_enemy_id") is not None]

        if friendly_fmts:
            # formation_id → 해당 편대 생존 기체 위치 목록
            state_by_fid: Dict[int, List[Dict]] = {}
            for s in states:
                if not s["is_alive"]:
                    continue
                fid = s.get("formation_id")
                if fid is not None:
                    state_by_fid.setdefault(fid, []).append(s)

            zones = []
            for fmt in friendly_fmts:
                f_fid = fmt["formation_id"]
                e_fid = fmt["paired_enemy_id"]
                f_alive = state_by_fid.get(f_fid, [])
                e_alive = state_by_fid.get(e_fid, [])
                if not f_alive and not e_alive:
                    continue
                combined = f_alive + e_alive
                clon = float(np.mean([s["lon"] for s in combined]))
                clat = float(np.mean([s["lat"] for s in combined]))
                f_base = fmt.get("base_name", "?")
                # 적군 편대 기지명
                e_fmt = next((f for f in formations if f["formation_id"] == e_fid), {})
                e_base = e_fmt.get("base_name", "?")
                zones.append({
                    "center_lon": clon,
                    "center_lat": clat,
                    "radius_km": 20.0,
                    "label": f"{f_base} vs {e_base}",
                })
            if zones:
                return zones

        # ── Fallback: flight_phase == 'combat' 기체 기반 ─────────────────
        combat_f = [s for s in states if s["team"] == "friendly"
                    and s.get("flight_phase") == "combat" and s["is_alive"]]
        combat_e = [s for s in states if s["team"] == "enemy"
                    and s.get("flight_phase") == "combat" and s["is_alive"]]
        if combat_f and combat_e:
            clon = float(np.mean([s["lon"] for s in combat_f + combat_e]))
            clat = float(np.mean([s["lat"] for s in combat_f + combat_e]))
            return [{"center_lon": clon, "center_lat": clat,
                     "radius_km": 20.0, "label": "교전구역"}]

        # ── Fallback: 20km 이내 근접 기체 쌍 ────────────────────────────
        zones = []
        alive_f = [s for s in states if s["team"] == "friendly" and s["is_alive"]]
        alive_e = [s for s in states if s["team"] == "enemy"    and s["is_alive"]]
        seen: set = set()
        for f in alive_f:
            for e in alive_e:
                dist = _haversine_km(f["lon"], f["lat"], e["lon"], e["lat"])
                if dist <= 20.0:
                    key = (round(f["lon"], 1), round(f["lat"], 1))
                    if key not in seen:
                        zones.append({
                            "center_lon": (f["lon"] + e["lon"]) / 2,
                            "center_lat": (f["lat"] + e["lat"]) / 2,
                            "radius_km": 20.0,
                            "label": f"교전구역 ({dist:.0f}km)",
                        })
                        seen.add(key)
        return zones

    # ------------------------------------------------------------------
    # 탭 1: 지도 생성 (matplotlib — JS/Plotly 버전 충돌 없이 PNG로 렌더링)
    # ------------------------------------------------------------------

    # 한반도 간략 윤곽선 (경위도)
    _PENINSULA = [
        # 남한 서해안
        (124.6,37.8),(125.1,37.7),(126.1,37.2),(126.3,36.6),
        (126.5,36.0),(126.3,35.5),(126.5,35.0),(127.0,34.5),
        # 남한 남해안·동해안
        (127.5,34.5),(128.5,34.8),(129.0,35.1),(129.3,36.0),
        (129.4,37.0),(129.5,38.0),
        # 휴전선 ~ 북한 동해안
        (129.6,38.6),(130.5,40.0),(130.6,41.5),(130.5,42.5),
        # 북한 북쪽 국경
        (129.0,42.5),(128.0,42.0),(126.5,42.0),(125.5,41.8),
        (124.5,40.5),(124.3,40.0),
        # 북한 서해안 → 시작점으로
        (124.5,39.5),(124.7,39.0),(124.6,38.5),(124.6,37.8),
    ]

    def _make_map_figure(self) -> plt.Figure:
        states = self._states()
        zones  = self._detect_combat_zones(states)

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#1a1a2e")

        # ── 한반도 윤곽 ──────────────────────────────────────────────
        px = [p[0] for p in self._PENINSULA]
        py = [p[1] for p in self._PENINSULA]
        ax.fill(px, py, color="#2d2d44", zorder=1)
        ax.plot(px, py, color="#6272a4", linewidth=0.8, zorder=2)

        # ── 휴전선 (38선 부근) ────────────────────────────────────────
        ax.axhline(38.3, color="#f38ba8", linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=3)
        ax.text(124.2, 38.4, "DMZ", color="#f38ba8", fontsize=7, alpha=0.8)

        # ── 교전 구역 ────────────────────────────────────────────────
        for zone in zones:
            lons, lats = _circle_coords(
                zone["center_lon"], zone["center_lat"], zone["radius_km"]
            )
            ax.fill(lons, lats, color="red", alpha=0.15, zorder=4)
            ax.plot(lons, lats, color="#ff5050", linewidth=1.5, zorder=4)

        # ── 기지 ─────────────────────────────────────────────────────
        for i, (name, info) in enumerate(FRIENDLY_BASES.items()):
            ax.plot(info["lon"], info["lat"], "s",
                    color="#1a6fd4", markersize=11,
                    markeredgecolor="#89b4fa", markeredgewidth=1.5, zorder=6)
            ax.text(info["lon"] + 0.05, info["lat"] + 0.05,
                    f"F-Base{i+1}", color="#89b4fa", fontsize=7.5,
                    fontweight="bold", zorder=7)

        for i, (name, info) in enumerate(ENEMY_BASES.items()):
            ax.plot(info["lon"], info["lat"], "s",
                    color="#c0392b", markersize=11,
                    markeredgecolor="#f38ba8", markeredgewidth=1.5, zorder=6)
            ax.text(info["lon"] + 0.05, info["lat"] + 0.05,
                    f"E-Base{i+1}", color="#f38ba8", fontsize=7.5,
                    fontweight="bold", zorder=7)

        # ── 항공기 ───────────────────────────────────────────────────
        for s in states:
            phase   = s.get("flight_phase", "unknown")
            marker  = _PHASE_MARKER.get(phase, "o")
            alive   = bool(s["is_alive"])
            if s["team"] == "friendly":
                color = "#00b4ff" if alive else "#7fb8d4"
            else:
                color = "#ff3030" if alive else "#d47f7f"
            msize = 10 if alive else 7

            if not alive:
                marker = "x"

            ax.plot(s["lon"], s["lat"], marker,
                    color=color, markersize=msize,
                    markeredgecolor="white" if alive else color,
                    markeredgewidth=0.5, zorder=8)

            # 기체 ID 레이블
            ax.text(s["lon"] + 0.04, s["lat"] + 0.04,
                    s["aircraft_uid"], color=color,
                    fontsize=6.5, zorder=9)

        # ── 축 꾸미기 ────────────────────────────────────────────────
        ax.set_xlim(124.0, 131.0)
        ax.set_ylim(34.5, 43.0)
        ax.set_xlabel("Lon (E)", color="#cdd6f4", fontsize=9)
        ax.set_ylabel("Lat (N)", color="#cdd6f4", fontsize=9)
        ax.tick_params(colors="#cdd6f4", labelsize=8)
        ax.grid(True, color="#45475a", linewidth=0.4, alpha=0.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

        # ── 범례 ─────────────────────────────────────────────────────
        legend_items = [
            Line2D([0],[0], marker="s", color="w", markerfacecolor="#1a6fd4",
                   markersize=9, label="Friendly Base", linestyle="None"),
            Line2D([0],[0], marker="s", color="w", markerfacecolor="#c0392b",
                   markersize=9, label="Enemy Base", linestyle="None"),
            Line2D([0],[0], marker="^", color="w", markerfacecolor="#00b4ff",
                   markersize=9, label="Friendly (alive)", linestyle="None"),
            Line2D([0],[0], marker="x", color="#7fb8d4",
                   markersize=9, label="Friendly (KIA)", linestyle="None"),
            Line2D([0],[0], marker="^", color="w", markerfacecolor="#ff3030",
                   markersize=9, label="Enemy (alive)", linestyle="None"),
            Line2D([0],[0], marker="x", color="#d47f7f",
                   markersize=9, label="Enemy (KIA)", linestyle="None"),
        ]
        legend = ax.legend(
            handles=legend_items, loc="upper right",
            facecolor="#2d2d44", edgecolor="#45475a",
            labelcolor="#cdd6f4", fontsize=8,
        )

        fig.tight_layout(pad=0.5)
        return fig

    def _render_map_b64(self) -> str:
        """matplotlib figure → base64 PNG 문자열."""
        fig = self._make_map_figure()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="#1e1e2e",
                    bbox_inches="tight", dpi=90)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _make_map_carrier(self) -> str:
        """타이머 갱신용: base64 데이터만 숨겨진 span 에 담아 전달.
        JS 가 이를 감지해 #map-img 의 src 만 교체 → 플리커 없음."""
        return f'<span id="map-b64">{self._render_map_b64()}</span>'

    # ------------------------------------------------------------------
    # 탭 1: 이벤트 로그 테이블
    # ------------------------------------------------------------------

    def _make_event_df(self) -> pd.DataFrame:
        events = self._events()
        if not events:
            return pd.DataFrame(columns=["스텝", "유형", "세부 내용", "LLM 결정", "처리여부"])

        rows = []
        for e in events:
            etype    = _EVENT_TYPE_KO.get(e.get("event_type", ""), e.get("event_type", "-"))
            details  = e.get("details_json", {})
            decision = e.get("llm_decision") or {}

            detail_str = ""
            if isinstance(details, dict):
                if "loss_ratio" in details:
                    detail_str = (
                        f"생존 {details.get('alive_friendly', '?')}/"
                        f"{details.get('total_friendly', '?')}대 "
                        f"({details['loss_ratio']*100:.0f}% 손실)"
                    )
                elif "aircraft_uid" in details:
                    detail_str = f"{details['aircraft_uid']} 무장 고갈"
                else:
                    detail_str = str(details)[:80]

            decision_str = ""
            if isinstance(decision, dict):
                action_ko = {
                    "rtb":             "전체 RTB",
                    "request_support": "지원 요청",
                    "continue":        "계속 임무",
                }.get(decision.get("action", ""), decision.get("action", "-"))
                decision_str = action_ko

            rows.append({
                "스텝":    e.get("step", "-"),
                "유형":    etype,
                "세부 내용": detail_str,
                "LLM 결정": decision_str,
                "처리여부":  "완료" if e.get("resolved") else "대기",
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 탭 2: 기체 상태 테이블
    # ------------------------------------------------------------------

    def _make_aircraft_df(self) -> pd.DataFrame:
        states = self._states()
        if not states:
            return pd.DataFrame(columns=[
                "UID", "팀", "생존", "기지",
                "고도(m)", "속도(m/s)", "미사일",
                "비행단계", "스텝",
            ])

        rows = []
        for s in sorted(states, key=lambda x: (x["team"], x["aircraft_uid"])):
            rows.append({
                "UID":      s["aircraft_uid"],
                "팀":       "아군" if s["team"] == "friendly" else "적군",
                "생존":     "✔" if s["is_alive"] else "✘",
                "기지":     s.get("base_name", "-"),
                "고도(m)":  f"{s.get('alt', 0):.0f}",
                "속도(m/s)": f"{s.get('speed_mps', 0):.0f}",
                "미사일":   s.get("missiles_left", 0),
                "비행단계": _PHASE_KO.get(s.get("flight_phase", "unknown"), "-"),
                "스텝":     s.get("step", "-"),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 탭 2: LLM 판단 로그
    # ------------------------------------------------------------------

    def _make_llm_df(self) -> pd.DataFrame:
        decisions = self._llm_decisions()
        if not decisions:
            return pd.DataFrame(columns=["스텝", "판단유형", "결정", "근거요약"])

        type_ko = {
            "dispatch":      "출격 결정",
            "major_loss":    "50% 손실 대응",
            "ammo_depleted": "무장 고갈 대응",
        }
        rows = []
        for d in decisions:
            try:
                out = json.loads(d.get("output_decision", "{}"))
            except Exception:
                out = {}

            action = out.get("action", "")
            dispatch_info = ""
            if d.get("decision_type") == "dispatch" and "friendly_dispatch" in out:
                pairs = out["friendly_dispatch"]
                dispatch_info = " | ".join(
                    f"{p.get('base','?')} → {p.get('oppose','?')}" for p in pairs
                )

            action_ko = {
                "rtb":             "RTB 명령",
                "request_support": "지원 요청",
                "continue":        "임무 지속",
            }.get(action, dispatch_info or action or "-")

            reasoning = (d.get("reasoning") or out.get("reasoning", "-"))[:120]

            rows.append({
                "스텝":    d.get("step", "-"),
                "판단유형": type_ko.get(d.get("decision_type", ""), d.get("decision_type", "-")),
                "결정":    action_ko,
                "근거요약": reasoning,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 상태 요약: 정적 골격 (최초 1회) + JS in-place 갱신용 메서드
    # ------------------------------------------------------------------

    @staticmethod
    def _make_status_skeleton() -> str:
        return """
<div class='status-box'>
  <div class='status-title'>📡 전투 현황 (스텝: <span id='s-step'>-</span>)</div>
  <table class='status-tbl'>
    <tr>
      <td class='blue-text'>🔵 아군</td>
      <td><div class='bar-wrap'><div class='bar-fill blue-bar' id='s-bar-f' style='width:0%'></div></div></td>
      <td class='cnt blue-text' id='s-cnt-f'>-/-</td>
    </tr>
    <tr>
      <td class='red-text'>🔴 적군</td>
      <td><div class='bar-wrap'><div class='bar-fill red-bar' id='s-bar-e' style='width:0%'></div></div></td>
      <td class='cnt red-text' id='s-cnt-e'>-/-</td>
    </tr>
  </table>
</div>"""

    # ------------------------------------------------------------------
    # 기체 현황 패널: 정적 골격 + JS ac-tbody 갱신용
    # ------------------------------------------------------------------

    @staticmethod
    def _make_aircraft_status_skeleton() -> str:
        return """
<div class='ac-status-wrapper'>
  <div class='ac-status-header'>✈ 기체 현황</div>
  <div class='ac-status-scroll'>
    <table class='ac-status-tbl'>
      <thead>
        <tr><th>UID</th><th>팀</th><th>상태</th><th>기지</th><th>미사일</th></tr>
      </thead>
      <tbody id='ac-tbody'>
        <tr><td colspan='5' class='no-event'>대기 중...</td></tr>
      </tbody>
    </table>
  </div>
</div>"""

    def _make_aircraft_status_rows(self) -> str:
        """<tbody> 내부 <tr> 행들 — JS 가 ac-tbody.innerHTML 에 삽입."""
        import html as _html
        states = self._states()
        if not states:
            return "<tr><td colspan='5' class='no-event'>데이터 없음</td></tr>"

        rows_html = ""
        for s in sorted(states, key=lambda x: (x["team"] != "friendly", x["aircraft_uid"])):
            uid = _html.escape(s["aircraft_uid"])
            if s["team"] == "friendly":
                team_str = "🔵 아군"
                team_cls = "blue-text"
            else:
                team_str = "🔴 적군"
                team_cls = "red-text"

            cause = s.get("death_cause", "alive")
            if s["is_alive"]:
                status_str = "생존"
                status_cls = "status-alive"
            elif cause == "crashed":
                status_str = "추락"
                status_cls = "status-crashed"
            else:
                status_str = "격추"
                status_cls = "status-dead"

            base = _html.escape(s.get("base_name", "-"))
            missiles = s.get("missiles_left", 0) if s["is_alive"] else "-"
            rows_html += (
                f"<tr>"
                f"<td class='{team_cls}'>{uid}</td>"
                f"<td class='{team_cls}'>{team_str}</td>"
                f"<td class='{status_cls}'>{status_str}</td>"
                f"<td>{base}</td>"
                f"<td style='text-align:center'>{missiles}</td>"
                f"</tr>\n"
            )
        return rows_html

    # ------------------------------------------------------------------
    # 이벤트 로그: 정적 골격 (최초 1회) + JS ev-tbody 갱신용
    # ------------------------------------------------------------------

    @staticmethod
    def _make_event_skeleton() -> str:
        return """
<div class='event-wrapper'>
  <div class='event-header'>🚨 이벤트 로그</div>
  <div class='event-scroll' id='ev-scroll'>
    <table class='event-tbl'>
      <thead>
        <tr>
          <th>스텝</th><th>유형</th><th>세부 내용</th>
          <th>LLM 결정</th><th>처리</th>
        </tr>
      </thead>
      <tbody id='ev-tbody'>
        <tr><td colspan='5' class='no-event'>이벤트 없음</td></tr>
      </tbody>
    </table>
  </div>
</div>"""

    # ------------------------------------------------------------------
    # 상태 요약 HTML (타이머마다 gr.HTML 직접 교체 — 폴백용으로 유지)
    # ------------------------------------------------------------------

    def _make_status_html(self) -> str:
        """현재 DB 데이터로 상태 박스 전체 HTML 생성."""
        states = self._states()
        alive_f = sum(1 for s in states if s["team"] == "friendly" and s["is_alive"])
        total_f = sum(1 for s in states if s["team"] == "friendly")
        alive_e = sum(1 for s in states if s["team"] == "enemy"    and s["is_alive"])
        total_e = sum(1 for s in states if s["team"] == "enemy")
        step    = max((s.get("step", 0) for s in states), default=0) if states else "-"
        bar_f   = int(alive_f / max(total_f, 1) * 100)
        bar_e   = int(alive_e / max(total_e, 1) * 100)
        return f"""
<div class='status-box'>
  <div class='status-title'>📡 전투 현황 (스텝: {step})</div>
  <table class='status-tbl'>
    <tr>
      <td class='blue-text'>🔵 아군</td>
      <td><div class='bar-wrap'><div class='bar-fill blue-bar' style='width:{bar_f}%'></div></div></td>
      <td class='cnt blue-text'>{alive_f}/{total_f}</td>
    </tr>
    <tr>
      <td class='red-text'>🔴 적군</td>
      <td><div class='bar-wrap'><div class='bar-fill red-bar' style='width:{bar_e}%'></div></div></td>
      <td class='cnt red-text'>{alive_e}/{total_e}</td>
    </tr>
  </table>
</div>"""

    # ------------------------------------------------------------------
    # 이벤트 로그 HTML (타이머마다 gr.HTML 직접 교체)
    # ------------------------------------------------------------------

    def _make_event_html(self) -> str:
        """이벤트 로그 전체 HTML 생성."""
        rows = self._make_event_rows()
        return f"""
<div class='event-wrapper'>
  <div class='event-header'>🚨 이벤트 로그</div>
  <div class='event-scroll' id='ev-scroll'>
    <table class='event-tbl'>
      <thead>
        <tr>
          <th>스텝</th><th>유형</th><th>세부 내용</th>
          <th>LLM 결정</th><th>처리</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""

    # LLM 판단 유형 한글명
    _LLM_DECISION_TYPE_KO = {
        "dispatch":                   "초기 출격 결정",
        "formation_assignment":       "편대 배정",
        "major_loss":                 "전력 50% 손실 대응",
        "ammo_depleted":              "무장 고갈 대응",
        "formation_ammo_depleted":    "편대 무장 고갈 대응",
        "formation_destroyed":        "편대 전멸 교체 결정",
        "enemy_formation_destroyed":  "적 전멸 후 재배치 결정",
    }

    def _make_event_rows(self) -> str:
        """
        <tbody> 내부 <tr> 행들 반환 — JS가 ev-tbody.innerHTML에 삽입.

        전술 이벤트(events 테이블) + LLM 판단(llm_decisions 테이블)을
        스텝 순서로 통합하여 표시.
        - 전술 이벤트: 기존 스타일 유지
        - LLM 판단: 보라색(event-llm) 별도 스타일로 구분
        """
        import html as _html

        # ── 전술 이벤트 행 수집 ───────────────────────────────────────
        events = self._events()
        rows: List[Dict] = []

        for e in events:
            etype   = _EVENT_TYPE_KO.get(e.get("event_type", ""), e.get("event_type", "-"))
            details = e.get("details_json", {})
            dec     = e.get("llm_decision") or {}

            if isinstance(details, dict):
                if "loss_ratio" in details:
                    detail_str = (
                        f"생존 {_html.escape(str(details.get('alive_friendly','?')))}/"
                        f"{_html.escape(str(details.get('total_friendly','?')))}대 "
                        f"({details['loss_ratio']*100:.0f}% 손실)"
                    )
                elif "aircraft_uid" in details:
                    detail_str = f"{_html.escape(str(details['aircraft_uid']))} 무장 고갈"
                elif "friendly_base" in details and "alive_friendly" in details and "enemy_base" in details and "alive_enemy" not in details:
                    # enemy_formation_destroyed 이벤트
                    detail_str = (
                        f"🏆 {_html.escape(str(details.get('friendly_base','?')))} 편대 "
                        f"→ {_html.escape(str(details.get('enemy_base','?')))} 전멸 "
                        f"(아군 {details.get('alive_friendly','?')}대 생존)"
                    )
                elif "friendly_base" in details and "alive_enemy" in details:
                    detail_str = (
                        f"{_html.escape(str(details.get('friendly_base','?')))} 편대 "
                        f"(잔여 적기 {details.get('alive_enemy','?')}대)"
                    )
                else:
                    detail_str = _html.escape(str(details)[:60])
            else:
                detail_str = _html.escape(str(details)[:60])

            action_ko = {
                "rtb":             "전체 RTB",
                "request_support": "지원 요청",
                "continue":        "임무 지속",
            }.get(dec.get("action", ""),
                  _html.escape(str(dec.get("action", "-"))) if dec else "-")

            resolved_badge = (
                "<span class='badge-ok'>완료</span>"
                if e.get("resolved")
                else "<span class='badge-wait'>대기</span>"
            )
            if "손실" in etype or ("전멸" in etype and "적" not in etype):
                type_class = "event-loss"
            elif "적 편대 전멸" in etype:
                type_class = "status-alive"   # 승리 → 초록색
            else:
                type_class = "event-ammo"

            rows.append({
                "step": e.get("step", 0),
                "html": (
                    f"<tr>"
                    f"<td>{e.get('step','-')}</td>"
                    f"<td class='{type_class}'>{_html.escape(etype)}</td>"
                    f"<td>{detail_str}</td>"
                    f"<td>{action_ko}</td>"
                    f"<td>{resolved_badge}</td>"
                    f"</tr>\n"
                ),
            })

        # ── LLM 판단 행 수집 ─────────────────────────────────────────
        for d in self._llm_decisions():
            dtype   = d.get("decision_type", "")
            type_ko = self._LLM_DECISION_TYPE_KO.get(dtype, dtype)
            step_v  = d.get("step", 0)

            # 출력 파싱
            try:
                out = json.loads(d.get("output_decision", "{}"))
            except Exception:
                out = {}

            # 세부 내용: 판단 유형별 요약
            if dtype == "dispatch":
                pairs = out.get("friendly_dispatch", [])
                detail_str = " | ".join(
                    f"{_html.escape(p.get('base','?'))} → {_html.escape(p.get('oppose','?'))}"
                    for p in pairs
                ) or "-"
            elif dtype == "formation_assignment":
                pairs = out.get("assignment", [])
                detail_str = " | ".join(
                    f"{_html.escape(p.get('friendly_base','?'))} ↔ {_html.escape(p.get('enemy_base','?'))}"
                    for p in pairs
                ) or "-"
            elif dtype == "formation_destroyed":
                chosen = _html.escape(str(out.get("base", "-")))
                detail_str = f"교체 기지: {chosen}"
            elif dtype == "enemy_formation_destroyed":
                action = out.get("action", "")
                if action == "engage":
                    target = _html.escape(str(out.get("target_pair_idx", "-")))
                    detail_str = f"재교전 → 적편대{target}"
                else:
                    detail_str = "기지 복귀 명령"
            else:
                detail_str = "-"

            # 결정 내용
            action = out.get("action", "")
            action_ko = {
                "rtb":             "RTB 명령",
                "request_support": "지원 요청",
                "continue":        "임무 지속",
                "engage":          "재교전 명령",
            }.get(action, "")
            if not action_ko:
                # dispatch / assignment 등은 detail에 이미 표현됨
                action_ko = _html.escape(str(out.get("base", action or "-")))

            # 근거 (최대 60자)
            reasoning = _html.escape(
                (d.get("reasoning") or out.get("reasoning", ""))[:60]
            )
            detail_full = f"{detail_str}<br><span style='color:#a6adc8;font-size:0.78rem'>{reasoning}</span>"

            rows.append({
                "step": step_v,
                "html": (
                    f"<tr class='ev-row-llm'>"
                    f"<td>{step_v}</td>"
                    f"<td class='event-llm'>🤖 {_html.escape(type_ko)}</td>"
                    f"<td>{detail_full}</td>"
                    f"<td>{action_ko}</td>"
                    f"<td><span class='badge-llm'>LLM</span></td>"
                    f"</tr>\n"
                ),
            })

        if not rows:
            return "<tr><td colspan='5' class='no-event'>이벤트 없음</td></tr>"

        # 스텝 오름차순 정렬 후 HTML 조합
        rows.sort(key=lambda r: r["step"])
        return "".join(r["html"] for r in rows)

    # ------------------------------------------------------------------
    # 데이터 페이로드: status + events 를 단일 JSON으로 묶어 반환
    # CSS hidden data-carrier(gr.HTML)에 실어 MutationObserver로 감지
    # → JS가 s-step / s-bar-f / ev-tbody 등 DOM 노드를 직접 수정 (in-place)
    # ------------------------------------------------------------------

    def _make_data_payload(self) -> str:
        import html as _html
        payload = json.dumps({
            "status": json.loads(self._make_status_data()),
            "events": self._make_event_rows(),
            "aircraft": self._make_aircraft_status_rows(),
        })
        # JSON 전체를 html.escape → <div> textContent로 안전하게 읽기
        return f'<div id="dyn-payload">{_html.escape(payload)}</div>'

    def _make_status_data(self) -> str:
        states = self._states()
        if not states:
            return json.dumps({"step": "-", "bar_f": 0, "bar_e": 0,
                               "cnt_f": "-/-", "cnt_e": "-/-"})
        alive_f = sum(1 for s in states if s["team"] == "friendly" and s["is_alive"])
        total_f = sum(1 for s in states if s["team"] == "friendly")
        alive_e = sum(1 for s in states if s["team"] == "enemy"    and s["is_alive"])
        total_e = sum(1 for s in states if s["team"] == "enemy")
        step    = max((s.get("step", 0) for s in states), default=0)
        return json.dumps({
            "step":  step,
            "bar_f": int(alive_f / max(total_f, 1) * 100),
            "bar_e": int(alive_e / max(total_e, 1) * 100),
            "cnt_f": f"{alive_f}/{total_f}",
            "cnt_e": f"{alive_e}/{total_e}",
        })

    # ------------------------------------------------------------------
    # 통합 갱신 콜백 — timer.tick 출력 2개: 지도 / data-carrier
    # ------------------------------------------------------------------

    def _refresh(self):
        return (
            self._make_map_carrier(),   # map-carrier → JS가 img.src 업데이트
            self._make_data_payload(),  # data-carrier → JS가 상태패널 업데이트
        )

    # ------------------------------------------------------------------
    # Gradio 앱 빌드 (단일 페이지)
    # ------------------------------------------------------------------

    def build(self) -> gr.Blocks:
        css = """
        /* ── 전체 배경 ── */
        body, .gradio-container { background:#1e1e2e !important; color:#cdd6f4; }
        footer { display:none !important; }

        /* ── 깜빡임 완전 제거: 모든 애니메이션·전환 비활성화 ── */
        *, *::before, *::after {
          animation: none !important;
          animation-duration: 0s !important;
          transition: none !important;
          transition-duration: 0s !important;
        }
        /* 예외: 생존율 막대 너비 전환만 유지 */
        .bar-fill { transition: width 0.4s ease !important; }
        /* 로딩 스피너 숨김 */
        .loader, .loader-wrap, svg.loader-spinner { display:none !important; }
        /* 로딩 중 opacity 강제 유지 */
        .block, .wrap, .block.generating, .block.pending,
        .wrap.generating, .wrap.pending { opacity:1 !important; }

        /* ── 지도: 영구 img 고정, carrier 숨김 ── */
        #map-container { min-height: 500px; }
        #map-container img { display:block; width:100%; height:auto; }
        #map-carrier { display:none !important; height:0 !important; overflow:hidden !important; }

        /* ── data-carrier: CSS로 숨김 (visible=False 대신 사용)
         * visible=False → Svelte 조건부 렌더링 → DOM 제거 → 업데이트 미수신
         * CSS display:none → DOM 유지 → Gradio 업데이트 수신 → MutationObserver 동작 */
        #data-carrier { display:none !important; height:0 !important;
                        overflow:hidden !important; margin:0 !important; padding:0 !important; }

        /* ── 헤더 ── */
        .header-md h1 { color:#89b4fa; margin-bottom:2px; }
        .header-md p  { color:#a6adc8; font-size:0.85rem; margin:0; }

        /* ── 상태 박스 ── */
        .status-box {
          background:#313244; border-radius:8px; padding:12px 14px;
          font-size:0.9rem; height:100%;
        }
        .status-title { font-weight:700; margin-bottom:8px; color:#cba6f7; }
        .status-tbl   { width:100%; border-collapse:collapse; border:1px solid #ffffff; }
        .status-tbl td { padding:4px 6px; vertical-align:middle; border:1px solid #ffffff; }
        .blue-text { color:#89b4fa; font-weight:600; white-space:nowrap; }
        .red-text  { color:#f38ba8; font-weight:600; white-space:nowrap; }
        .cnt { text-align:right; font-weight:700; white-space:nowrap; width:52px; }
        .bar-wrap  {
          background:#45475a; border-radius:4px;
          height:10px; width:140px; overflow:hidden;
        }
        .bar-fill  { height:100%; border-radius:4px; transition:width 0.4s; }
        .blue-bar  { background:#89b4fa; }
        .red-bar   { background:#f38ba8; }

        /* ── 범례 박스 ── */
        .legend-md { background:#313244; border-radius:8px; padding:10px 14px; font-size:0.82rem; color:#ffffff; }
        .legend-md table { font-size:0.82rem; color:#ffffff; }
        .legend-md td { color:#ffffff; }
        .legend-md b { color:#ffffff; }

        /* ── 기체 현황 패널 ── */
        .ac-status-wrapper { background:#313244; border-radius:8px; overflow:hidden; margin-top:8px; }
        .ac-status-header  {
          background:#45475a; padding:6px 14px;
          font-weight:700; color:#a6e3a1; font-size:0.88rem;
        }
        .ac-status-scroll  { max-height:220px; overflow-y:auto; padding:4px; }
        .ac-status-tbl     { width:100%; border-collapse:collapse; font-size:0.80rem; }
        .ac-status-tbl thead tr { background:#1e1e2e; position:sticky; top:0; z-index:2; }
        .ac-status-tbl th  { padding:5px 6px; color:#a6adc8; font-weight:600;
                             border-bottom:1px solid #45475a; white-space:nowrap; }
        .ac-status-tbl td  { padding:4px 6px; border-bottom:1px solid #313244;
                             color:#cdd6f4; white-space:nowrap; }
        .ac-status-tbl tr:last-child td { border-bottom:none; }
        .ac-status-tbl tr:hover td { background:#383850; }
        .status-alive   { color:#a6e3a1; font-weight:700; }
        .status-dead    { color:#f38ba8; font-weight:700; }
        .status-crashed { color:#fab387; font-weight:700; }

        /* ── 이벤트 로그 ── */
        .event-wrapper { background:#313244; border-radius:8px; overflow:hidden; }
        .event-header  {
          background:#45475a; padding:8px 14px;
          font-weight:700; color:#f9e2af; font-size:0.9rem;
        }
        .event-scroll  { max-height:260px; overflow-y:auto; padding:4px; }
        .event-tbl     { width:100%; border-collapse:collapse; font-size:0.82rem; }
        .event-tbl thead tr  { background:#1e1e2e; position:sticky; top:0; z-index:2; }
        .event-tbl th  { padding:6px 8px; color:#a6adc8; font-weight:600; border-bottom:1px solid #45475a; }
        .event-tbl td  { padding:5px 8px; border-bottom:1px solid #313244; color:#cdd6f4; }
        .event-tbl tr:last-child td { border-bottom:none; }
        .event-tbl tr:hover td { background:#383850; }
        .event-loss  { color:#f38ba8; font-weight:600; }
        .event-ammo  { color:#fab387; font-weight:600; }
        .event-llm   { color:#cba6f7; font-weight:600; }
        .no-event    { text-align:center; color:#585b70; padding:16px; }
        .badge-ok    {
          background:#a6e3a1; color:#1e1e2e; border-radius:4px;
          padding:1px 6px; font-size:0.75rem; font-weight:700;
        }
        .badge-wait  {
          background:#f9e2af; color:#1e1e2e; border-radius:4px;
          padding:1px 6px; font-size:0.75rem; font-weight:700;
        }
        .badge-llm   {
          background:#cba6f7; color:#1e1e2e; border-radius:4px;
          padding:1px 6px; font-size:0.75rem; font-weight:700;
        }
        .ev-row-llm td { background:#2a2a3e !important; }
        .ev-row-llm:hover td { background:#32324a !important; }

        /* ── 시뮬레이션 시작 버튼 ── */
        #start-sim-btn {
          width:100%; margin-top:8px;
          background:#a6e3a1 !important; color:#1e1e2e !important;
          border:none !important; border-radius:8px !important;
          font-size:0.95rem !important; font-weight:700 !important;
          padding:10px 0 !important; cursor:pointer !important;
          transition:background 0.2s;
        }
        #start-sim-btn:hover { background:#94d3a2 !important; }
        #start-sim-btn:disabled,
        #start-sim-btn[disabled] {
          background:#45475a !important; color:#a6adc8 !important;
          cursor:not-allowed !important;
        }
        """

        init_js = """
() => {
  /* ── 이벤트 로그 스크롤 초기값 ── */
  if (sessionStorage.getItem('evAtBottom') === null) {
    sessionStorage.setItem('evAtBottom', '1');
  }

  /* ── 상태·이벤트 in-place 업데이트 ─────────────────────────────────
   * gr.Timer → data_carrier(gr.HTML, CSS hidden) innerHTML 갱신
   * → MutationObserver 감지 → 보이는 DOM 노드만 직접 수정
   *
   * data-carrier 는 visible=False 가 아닌 CSS display:none 으로 숨김:
   *   visible=False → Svelte {#if} → DOM 제거 → 업데이트 미수신
   *   CSS hidden   → DOM 유지    → Gradio 업데이트 수신 → Observer 동작
   */

  /* 이벤트 스크롤 리스너 (최초 1회) */
  var _evScrollBound = false;
  function _bindEvScroll() {
    if (_evScrollBound) return;
    var el = document.getElementById('ev-scroll');
    if (!el) return;
    _evScrollBound = true;
    el.addEventListener('scroll', function() {
      sessionStorage.setItem('evScroll', el.scrollTop);
      var atBot = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      sessionStorage.setItem('evAtBottom', atBot ? '1' : '0');
    }, {passive: true});
  }

  function _applyPayload() {
    var el = document.getElementById('dyn-payload');
    if (!el) return;
    try {
      var data = JSON.parse(el.textContent);

      /* 상태 바 in-place 업데이트 (DOM 재생성 없음 → CSS transition 유지) */
      var s = data.status || {};
      var stepEl = document.getElementById('s-step');
      var barF   = document.getElementById('s-bar-f');
      var barE   = document.getElementById('s-bar-e');
      var cntF   = document.getElementById('s-cnt-f');
      var cntE   = document.getElementById('s-cnt-e');
      if (stepEl) stepEl.textContent = s.step !== undefined ? s.step : '-';
      if (barF)   barF.style.width   = (s.bar_f || 0) + '%';
      if (barE)   barE.style.width   = (s.bar_e || 0) + '%';
      if (cntF)   cntF.textContent   = s.cnt_f || '-/-';
      if (cntE)   cntE.textContent   = s.cnt_e || '-/-';

      /* 이벤트 tbody in-place 업데이트 (스크롤 위치 보존) */
      var tbody = document.getElementById('ev-tbody');
      if (tbody && data.events !== undefined) {
        tbody.innerHTML = data.events;
        _bindEvScroll();
        var scroll = document.getElementById('ev-scroll');
        if (scroll) {
          var atBot = sessionStorage.getItem('evAtBottom') !== '0';
          if (atBot) scroll.scrollTop = scroll.scrollHeight;
          else {
            var saved = sessionStorage.getItem('evScroll');
            if (saved !== null) scroll.scrollTop = parseInt(saved, 10);
          }
        }
      }

      /* 기체 현황 tbody in-place 업데이트 */
      var acTbody = document.getElementById('ac-tbody');
      if (acTbody && data.aircraft !== undefined) {
        acTbody.innerHTML = data.aircraft;
      }
    } catch(e) {}
  }

  /* data-carrier(gr.HTML, CSS hidden)가 DOM에 나타나면 Observer 연결 */
  function _setupCarrier() {
    var wrapper = document.getElementById('data-carrier');
    if (!wrapper) return false;
    new MutationObserver(_applyPayload).observe(
      wrapper, {childList: true, subtree: true, characterData: true}
    );
    return true;
  }

  if (!_setupCarrier()) {
    var _cChk = setInterval(function() {
      if (_setupCarrier()) clearInterval(_cChk);
    }, 200);
  }

  /* ── 지도 플리커 제거: map-carrier 변경 감지 → img.src 만 교체 ── */
  /* img 요소는 DOM 에 영구 고정, src 속성만 바꾸므로 빈 화면이 없음    */
  (function() {
    function _setupMapCarrier() {
      var carrier = document.getElementById('map-carrier');
      if (!carrier) { setTimeout(_setupMapCarrier, 300); return; }
      function _applyMap() {
        var span = document.getElementById('map-b64');
        if (!span) return;
        var b64 = span.textContent.trim();
        if (!b64) return;
        var img = document.getElementById('map-img');
        if (img) img.src = 'data:image/png;base64,' + b64;
      }
      new MutationObserver(_applyMap).observe(
        carrier, { childList: true, subtree: true, characterData: true }
      );
    }
    _setupMapCarrier();
  })();

  /* ── 스크롤 위치 보존 ── */
  (function() {
    var _savedScroll = 0;
    window.addEventListener('scroll', function() {
      _savedScroll = window.scrollY;
    }, { passive: true });
    function _watchMap() {
      var el = document.getElementById('map-carrier');
      if (!el) { setTimeout(_watchMap, 300); return; }
      new MutationObserver(function() {
        requestAnimationFrame(function() {
          window.scrollTo({ top: _savedScroll, behavior: 'instant' });
        });
      }).observe(el, { childList: true, subtree: true });
    }
    _watchMap();
  })();
}
"""
        # Gradio 버전별로 theme/css/js 허용 위치가 다름:
        #   < 6.0 : Blocks(theme=, css=, js=)
        #   6.0+  : launch(theme=, css=, js=)
        # → inspect 로 각 시그니처를 확인해 런타임에 자동 분배
        import inspect
        theme = gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
        )
        ui_kwargs = {"theme": theme, "css": css, "js": init_js}
        _blocks_params = inspect.signature(gr.Blocks.__init__).parameters
        _launch_params = inspect.signature(gr.Blocks.launch).parameters

        blocks_extra = {k: v for k, v in ui_kwargs.items() if k in _blocks_params}
        self._launch_ui  = {k: v for k, v in ui_kwargs.items() if k in _launch_params and k not in blocks_extra}

        with gr.Blocks(title="한반도 전술 공중전 시뮬레이터", **blocks_extra) as demo:

            # ── 헤더 ──────────────────────────────────────────────────
            gr.Markdown(
                "# 🛩️ 한반도 전술 공중전 시뮬레이터\n"
                "**EXAONE-3.5 LLM 지휘관 | JSBSim 기반 물리 시뮬레이션 | 실시간 DB 동기화**",
                elem_classes=["header-md"],
            )

            # ── 메인 행: 지도 + 우측 패널 ─────────────────────────────
            with gr.Row(equal_height=True):
                # 지도 — 영구 <img id="map-img"> + 숨겨진 map-carrier
                # 타이머는 carrier 에 base64 만 전달, JS 가 img.src 만 교체
                # → DOM 요소 교체 없음 → 플리커 없음
                with gr.Column(scale=3, min_width=420):
                    initial_b64 = self._render_map_b64()
                    gr.HTML(
                        value=(
                            f'<img id="map-img" '
                            f'src="data:image/png;base64,{initial_b64}" '
                            f'style="width:100%;height:auto;display:block;" />'
                        ),
                        elem_id="map-container",
                    )
                    map_carrier = gr.HTML(
                        value="", elem_id="map-carrier",
                    )

                # 우측 패널: 상태 요약(정적 골격, JS가 in-place 갱신) + 기체 현황 + 범례
                with gr.Column(scale=1, min_width=220):
                    gr.HTML(
                        value=self._make_status_skeleton(),
                        elem_id="status-panel",
                    )

                    gr.HTML(
                        value=self._make_aircraft_status_skeleton(),
                        elem_id="aircraft-status-panel",
                    )

                    gr.HTML(
                        """
<div class='legend-md'>
  <b>📋 범례</b>
  <table>
    <tr><td>🔵▲</td><td>아군 기체 생존</td></tr>
    <tr><td>🔵✕</td><td>아군 기체 손실</td></tr>
    <tr><td>🔴▲</td><td>적군 기체 생존</td></tr>
    <tr><td>🔴✕</td><td>적군 기체 손실</td></tr>
    <tr><td>✈🔵</td><td>아군 기지</td></tr>
    <tr><td>✈🔴</td><td>적군 기지</td></tr>
    <tr><td>⭕</td><td>교전구역 20km</td></tr>
  </table>
  <hr style='border-color:#45475a;margin:6px 0'>
  <b>비행 단계</b><br>
  접근▲ · 교전▲ · RTB▽ · 재장착● · 완료✕
</div>
""",
                        elem_classes=["legend-md"],
                    )

                    start_btn = gr.Button(
                        value="▶ 시뮬레이션 시작",
                        elem_id="start-sim-btn",
                        interactive=True,
                    )

            # ── 이벤트 로그 (정적 골격, JS가 ev-tbody만 갱신) ─────────
            gr.HTML(
                value=self._make_event_skeleton(),
                elem_id="event-log",
            )

            # ── data-carrier: CSS hidden gr.HTML ─────────────────────
            # visible=False 대신 CSS #data-carrier{display:none}으로 숨김
            # → DOM 유지 → Gradio 타이머 업데이트 수신 → MutationObserver 동작
            data_carrier = gr.HTML(value="", elem_id="data-carrier")

            # ── 자동 갱신 (gr.Timer) ──────────────────────────────────
            # 출력 2개: 지도 / data-carrier(JSON payload)
            timer = gr.Timer(value=self.refresh_interval)
            timer.tick(
                fn=self._refresh,
                outputs=[map_carrier, data_carrier],
            )

            def _on_start_click():
                if not self._sim_started:
                    self._sim_started = True
                    if self.start_callback:
                        import threading
                        t = threading.Thread(
                            target=self.start_callback, daemon=True, name="SimThread"
                        )
                        t.start()
                return gr.Button(value="⏳ 시뮬레이션 실행 중...", interactive=False)

            start_btn.click(fn=_on_start_click, inputs=[], outputs=[start_btn])

        return demo

    def launch(self, server_port: int = 7860, share: bool = False, **kwargs):
        demo = self.build()
        demo.launch(
            server_name="0.0.0.0",
            server_port=server_port,
            share=share,
            **self._launch_ui,
            **kwargs,
        )
