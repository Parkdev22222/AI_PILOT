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

import json
import math
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .combat_db import CombatDB
from .llm_commander import ENEMY_BASES, FRIENDLY_BASES

# ── 기지 정보 (고정 마커) ────────────────────────────────────────────────────

_ALL_BASES = {
    **{k: {**v, "team": "enemy"}   for k, v in ENEMY_BASES.items()},
    **{k: {**v, "team": "friendly"} for k, v in FRIENDLY_BASES.items()},
}

# 비행 단계 → 마커 모양
_PHASE_SYMBOL = {
    "approach":  "triangle-up",
    "combat":    "triangle-up",
    "rtb_loss":  "triangle-down",
    "reload":    "circle",
    "returning": "triangle-up",
    "support":   "triangle-up",
    "done":      "x",
    "unknown":   "circle",
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
    "major_loss":     "전력 50% 손실",
    "ammo_depleted":  "무장 고갈",
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
    ):
        self.db = CombatDB(db_path)
        self.sim_id = sim_id
        self.refresh_interval = refresh_interval

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
        flight_phase == 'combat' 인 기체들의 편대쌍 중심을 교전 구역으로 반환.
        [{center_lon, center_lat, radius_km, label}, …]
        """
        if not states:
            return []

        combat_friendly = [
            s for s in states
            if s["team"] == "friendly"
            and s.get("flight_phase") == "combat"
            and s["is_alive"]
        ]
        combat_enemy = [
            s for s in states
            if s["team"] == "enemy"
            and s.get("flight_phase") == "combat"
            and s["is_alive"]
        ]

        if not combat_friendly or not combat_enemy:
            # 거리 기반 fallback: 20km 이내 pair
            zones = []
            alive_f = [s for s in states if s["team"] == "friendly" and s["is_alive"]]
            alive_e = [s for s in states if s["team"] == "enemy"    and s["is_alive"]]
            seen = set()
            for f in alive_f:
                for e in alive_e:
                    dist = _haversine_km(f["lon"], f["lat"], e["lon"], e["lat"])
                    if dist <= 20.0:
                        key = (round(f["lon"], 2), round(f["lat"], 2))
                        if key not in seen:
                            clon = (f["lon"] + e["lon"]) / 2
                            clat = (f["lat"] + e["lat"]) / 2
                            zones.append({
                                "center_lon": clon, "center_lat": clat,
                                "radius_km": 20.0,
                                "label": f"교전구역 ({dist:.0f}km)",
                            })
                            seen.add(key)
            return zones

        clon = float(np.mean([s["lon"] for s in combat_friendly + combat_enemy]))
        clat = float(np.mean([s["lat"] for s in combat_friendly + combat_enemy]))
        return [{"center_lon": clon, "center_lat": clat,
                 "radius_km": 20.0, "label": "교전구역"}]

    # ------------------------------------------------------------------
    # 탭 1: 지도 생성
    # ------------------------------------------------------------------

    def _make_map_figure(self) -> go.Figure:
        states = self._states()
        zones  = self._detect_combat_zones(states)

        fig = go.Figure()

        # ── Trace 0: 교전 구역 (항상 단일 trace, None 구분자로 여러 원 연결) ──
        # go.Scattergeo는 오프라인 동작 (Plotly 내장 지리 데이터 사용)
        zone_lons: List[Optional[float]] = []
        zone_lats: List[Optional[float]] = []
        for zone in zones:
            lons, lats = _circle_coords(
                zone["center_lon"], zone["center_lat"], zone["radius_km"]
            )
            if zone_lons:
                zone_lons.append(None)
                zone_lats.append(None)
            zone_lons.extend(lons)
            zone_lats.extend(lats)
        fig.add_trace(go.Scattergeo(
            lon=zone_lons, lat=zone_lats,
            mode="lines",
            line=dict(color="rgba(255,80,80,0.9)", width=2),
            fill="toself",
            fillcolor="rgba(255,0,0,0.15)",
            name="교전구역",
            hoverinfo="skip",
            showlegend=bool(zones),
        ))

        # ── Trace 1: 아군 기지 (항상 고정) ───────────────────────────
        fb = list(FRIENDLY_BASES.items())
        fig.add_trace(go.Scattergeo(
            lon=[v["lon"] for _, v in fb],
            lat=[v["lat"] for _, v in fb],
            mode="markers+text",
            marker=dict(size=14, color="#1a6fd4", symbol="square",
                        line=dict(color="#89b4fa", width=2)),
            text=["✈ " + n for n, _ in fb],
            textposition="top right",
            textfont=dict(size=10, color="#89b4fa"),
            name="아군 기지",
            hovertext=[f"아군 기지: {n}" for n, _ in fb],
            hoverinfo="text",
            showlegend=True,
        ))

        # ── Trace 2: 적군 기지 (항상 고정) ───────────────────────────
        eb = list(ENEMY_BASES.items())
        fig.add_trace(go.Scattergeo(
            lon=[v["lon"] for _, v in eb],
            lat=[v["lat"] for _, v in eb],
            mode="markers+text",
            marker=dict(size=14, color="#c0392b", symbol="square",
                        line=dict(color="#f38ba8", width=2)),
            text=["✈ " + n for n, _ in eb],
            textposition="top right",
            textfont=dict(size=10, color="#f38ba8"),
            name="적군 기지",
            hovertext=[f"적군 기지: {n}" for n, _ in eb],
            hoverinfo="text",
            showlegend=True,
        ))

        # 항공기 분류
        alive_f = [s for s in states if s["team"] == "friendly" and     s["is_alive"]]
        dead_f  = [s for s in states if s["team"] == "friendly" and not s["is_alive"]]
        alive_e = [s for s in states if s["team"] == "enemy"    and     s["is_alive"]]
        dead_e  = [s for s in states if s["team"] == "enemy"    and not s["is_alive"]]

        def _hover(s: Dict) -> str:
            return (
                f"<b>{s['aircraft_uid']}</b><br>"
                f"팀: {'아군' if s['team']=='friendly' else '적군'}<br>"
                f"고도: {s.get('alt',0):.0f}m<br>"
                f"속도: {s.get('speed_mps',0):.0f}m/s<br>"
                f"미사일: {s.get('missiles_left',0)}발<br>"
                f"단계: {_PHASE_KO.get(s.get('flight_phase','unknown'),'-')}<br>"
                f"기지: {s.get('base_name','-')}"
            )

        # ── Trace 3·4·5·6: 항공기 (빈 배열이어도 항상 4개 trace 유지) ──
        # go.Scattergeo: symbol 배열 지원 → 비행 단계별 심볼 적용 (오프라인 OK)
        # 생존 기체: _PHASE_SYMBOL 딕셔너리로 단계별 심볼 차등 표시
        # 손실 기체: 고정 "x" 심볼
        for group, color, label, sz, use_phase_symbol in [
            (alive_f, "#00b4ff", "아군 (생존)", 14, True),
            (dead_f,  "#7fb8d4", "아군 (손실)",  9, False),
            (alive_e, "#ff3030", "적군 (생존)", 14, True),
            (dead_e,  "#d47f7f", "적군 (손실)",  9, False),
        ]:
            if use_phase_symbol:
                symbols = [
                    _PHASE_SYMBOL.get(s.get("flight_phase", "unknown"), "circle")
                    for s in group
                ]
            else:
                symbols = "x"
            fig.add_trace(go.Scattergeo(
                lon=[s["lon"] for s in group],
                lat=[s["lat"] for s in group],
                mode="markers",
                marker=dict(size=sz, color=color, symbol=symbols),
                name=label,
                hovertext=[_hover(s) for s in group],
                hoverinfo="text" if group else "skip",
                showlegend=True,
            ))

        # ── 지도 레이아웃 (완전 오프라인, Plotly 내장 데이터) ──────────
        # scope 없이 lonaxis/lataxis.range 로만 뷰 제어:
        # scope="asia" + 좁은 range 조합이 일부 Plotly 버전에서 흰 화면 유발
        fig.update_layout(
            geo=dict(
                projection_type="mercator",
                showland=True,       landcolor="#2d2d44",
                showocean=True,      oceancolor="#1a1a2e",
                showcoastlines=True, coastlinecolor="#6272a4",
                showcountries=True,  countrycolor="#6272a4",
                showlakes=False,     showrivers=False,
                bgcolor="#1e1e2e",
                lonaxis=dict(range=[124.0, 131.0]),
                lataxis=dict(range=[34.5, 43.0]),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
            legend=dict(
                bgcolor="rgba(30,30,46,0.8)",
                bordercolor="#45475a",
                borderwidth=1,
                font=dict(color="#cdd6f4", size=11),
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
            ),
            height=650,
            uirevision="map",  # 항상 동일 → Plotly.react() 시 줌/패닝 보존
        )
        return fig

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

    def _make_event_rows(self) -> str:
        """<tbody> 내부 <tr> 행들만 반환 — JS가 ev-tbody.innerHTML에 삽입."""
        import html as _html
        events = self._events()
        if not events:
            return "<tr><td colspan='5' class='no-event'>이벤트 없음</td></tr>"

        rows_html = ""
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
                else:
                    detail_str = _html.escape(str(details)[:60])
            else:
                detail_str = _html.escape(str(details)[:60])

            action_ko = {
                "rtb":             "전체 RTB",
                "request_support": "지원 요청",
                "continue":        "임무 지속",
            }.get(dec.get("action", ""), _html.escape(dec.get("action", "-")) if dec else "-")

            resolved_badge = (
                "<span class='badge-ok'>완료</span>"
                if e.get("resolved")
                else "<span class='badge-wait'>대기</span>"
            )
            type_class = "event-loss" if "손실" in etype else "event-ammo"

            rows_html += (
                f"<tr>"
                f"<td>{e.get('step','-')}</td>"
                f"<td class='{type_class}'>{_html.escape(etype)}</td>"
                f"<td>{detail_str}</td>"
                f"<td>{action_ko}</td>"
                f"<td>{resolved_badge}</td>"
                f"</tr>\n"
            )
        return rows_html

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
    # 지도 HTML 생성 — gr.HTML 컴포넌트용
    # ------------------------------------------------------------------
    # Gradio 6.x 가 번들하는 Plotly.js 는 구버전이라 Scattergeo 가 흰 화면이 됨.
    # Plotly Python 6.x 는 Plotly.js 3.x 를 사용하므로, to_html() 로 직접 embed.
    # include_plotlyjs='cdn' → 첫 로드 후 브라우저 캐시 → 이후 갱신은 ~7KB 만 전송.
    # Plotly.react() 사용 → 동일 div 재활용 시 줌/패닝 상태 유지.
    # ------------------------------------------------------------------

    def _make_map_html(self) -> str:
        fig = self._make_map_figure()
        fig_json = fig.to_json()
        return f"""<div id="tactical-map" style="height:650px;width:100%;background:#1e1e2e;"></div>
<script>(function(){{
  function render(){{
    var el=document.getElementById('tactical-map');
    if(!el||!window.Plotly){{setTimeout(render,100);return;}}
    var fig={fig_json};
    Plotly.react(el,fig.data,fig.layout,{{responsive:true,displayModeBar:false}});
  }}
  render();
}})();</script>"""

    # ------------------------------------------------------------------
    # 통합 갱신 콜백 — timer.tick 출력 2개: 지도 HTML / data-carrier
    # ------------------------------------------------------------------

    def _refresh(self):
        return (
            self._make_map_html(),
            self._make_data_payload(),  # CSS hidden gr.HTML → MutationObserver → in-place DOM
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
        .status-tbl   { width:100%; border-collapse:collapse; }
        .status-tbl td { padding:4px 6px; vertical-align:middle; }
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
        .legend-md { background:#313244; border-radius:8px; padding:10px 14px; font-size:0.82rem; }
        .legend-md table { font-size:0.82rem; }

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
        .no-event    { text-align:center; color:#585b70; padding:16px; }
        .badge-ok    {
          background:#a6e3a1; color:#1e1e2e; border-radius:4px;
          padding:1px 6px; font-size:0.75rem; font-weight:700;
        }
        .badge-wait  {
          background:#f9e2af; color:#1e1e2e; border-radius:4px;
          padding:1px 6px; font-size:0.75rem; font-weight:700;
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
        # Plotly.js 3.x CDN 로드 — Gradio 번들 버전(구버전)을 우회해 Scattergeo 정상 렌더링
        plotly_cdn = (
            '<script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>'
        )
        ui_kwargs = {"theme": theme, "css": css, "js": init_js, "head": plotly_cdn}
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
                # 지도 — gr.HTML + Plotly.js CDN (Gradio 번들 구버전 우회)
                with gr.Column(scale=4, min_width=580):
                    map_plot = gr.HTML(
                        value=self._make_map_html(),
                    )

                # 우측 패널: 상태 요약(정적 골격, JS가 in-place 갱신) + 범례
                with gr.Column(scale=1, min_width=220):
                    gr.HTML(
                        value=self._make_status_skeleton(),
                        elem_id="status-panel",
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
                outputs=[map_plot, data_carrier],
            )

        return demo

    def launch(self, server_port: int = 7860, share: bool = False, **kwargs):
        demo = self.build()
        demo.launch(
            server_name="0.0.0.0",
            server_port=server_port,
            share=share,
            **self._launch_ui,   # 버전별로 해당되는 theme/css/js 만 전달
            **kwargs,
        )
