"""
tactical_controller.py
=======================
한반도 공중전 전술 시뮬레이터 메인 컨트롤러.

실행 순서
---------
1. 적군 기지 랜덤 선택 (원산 / 평양 / 순천 중 N개)
2. LLM(EXAONE-3.5)이 아군 기지 N개 선택 및 편대-적 대응 결정
3. TacticalCombatEnv 생성 (선택된 기지 초기 위치 전달)
4. 접근 단계 (>20 km): 오토파일럿
5. 교전 단계 (<20 km): RL 정책
6. 이벤트 감지 → LLM 판단 → 처리
   - 50% 손실: 전체 RTB 또는 지원 편대 스폰
   - 무장 고갈: 개별 RTB + 재장착 + 복귀

Usage
-----
    python -m tactical_system.tactical_controller \
        --ego_policy  <path/to/ego_actor.pt> \
        --enm_policy  <path/to/enm_actor.pt> \
        --llm_model   LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
        --db_path     combat_simulation.db \
        --max_steps   3000 \
        --render
"""

import argparse
import logging
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── 경로 설정 ────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAC_ROOT  = os.path.join(_REPO_ROOT, "external", "CloseAirCombat")
for p in [_REPO_ROOT, _CAC_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tactical_system.combat_db import CombatDB
from tactical_system.llm_commander import (
    ENEMY_BASES,
    FRIENDLY_BASES,
    LLMCommander,
)
from tactical_system.tactical_combat_env import (
    APPROACH_DISTANCE_M,
    CRUISE_ALT_M,
    CRUISE_SPEED_FPS,
    SUPPORT_SPAWN_ALT_M,
    TacticalCombatEnv,
    _bearing_rad,
    _haversine_m,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tactical_controller")


# ─────────────────────────────────────────────────────────────────────────────
# 기지 → JSBSim 초기 상태 변환
# ─────────────────────────────────────────────────────────────────────────────

def _base_to_init_state(
    base_lon: float,
    base_lat: float,
    heading_deg: float,
    alt_m: float = CRUISE_ALT_M,
    u_fps: float = CRUISE_SPEED_FPS,
) -> Dict:
    return {
        "ic_long_gc_deg":  base_lon,
        "ic_lat_geod_deg": base_lat,
        "ic_h_sl_ft":      alt_m * 3.28084,
        "ic_psi_true_deg": heading_deg % 360,
        "ic_u_fps":        u_fps,
        "ic_v_fps":        0.0,
        "ic_w_fps":        0.0,
        "ic_p_rad_sec":    0.0,
        "ic_q_rad_sec":    0.0,
        "ic_r_rad_sec":    0.0,
        "ic_roc_fpm":      0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 가장 가까운 아군 기지 탐색
# ─────────────────────────────────────────────────────────────────────────────

def _closest_friendly_base(
    combat_lon: float,
    combat_lat: float,
    exclude: Optional[List[str]] = None,
) -> Optional[Dict]:
    """전투 지역과 가장 가까운 미사용 아군 기지 반환."""
    exclude = exclude or []
    best_name, best_dist = None, float("inf")
    for name, info in FRIENDLY_BASES.items():
        if name in exclude:
            continue
        d = _haversine_m(info["lon"], info["lat"], combat_lon, combat_lat)
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_name is None:
        return None
    return {"name": best_name, **FRIENDLY_BASES[best_name]}


# ─────────────────────────────────────────────────────────────────────────────
# 메인 컨트롤러
# ─────────────────────────────────────────────────────────────────────────────

class TacticalController:
    """전체 전술 시뮬레이션 조율."""

    def __init__(
        self,
        ego_policy_path: str = "",
        enm_policy_path: str = "",
        llm_model_id: str = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        db_path: str = "combat_simulation.db",
        max_steps: int = 3000,
        render: bool = False,
        render_path: str = "tactical_combat.txt.acmi",
        device: str = "cpu",
        llm_device: str = "auto",
        seed: int = 42,
        num_enemy_formations: Optional[int] = None,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.ego_policy_path = ego_policy_path
        self.enm_policy_path = enm_policy_path
        self.max_steps = max_steps
        self.render = render
        self.render_path = render_path
        self.device = device

        # ── DB ──────────────────────────────────────────────────────────
        self.db = CombatDB(db_path)
        logger.info(f"DB 경로: {db_path}")

        # ── 1단계: 적군 출격 기지 랜덤 선택 ──────────────────────────────
        all_enemy_bases = list(ENEMY_BASES.keys())
        n = (
            num_enemy_formations
            if num_enemy_formations is not None
            else random.randint(1, len(all_enemy_bases))
        )
        self.enemy_bases_selected: List[str] = random.sample(all_enemy_bases, n)
        logger.info(f"적군 출격 기지 ({n}개): {self.enemy_bases_selected}")

        # 전장 중심 (DMZ 근방)
        self.battle_field_center = (127.0, 38.5, 0.0)

        # ── DB 시뮬레이션 레코드 생성 ─────────────────────────────────────
        self.sim_id = self.db.create_simulation(
            num_friendly=n,
            num_enemy=n,
            center_lon=self.battle_field_center[0],
            center_lat=self.battle_field_center[1],
        )
        logger.info(f"시뮬레이션 ID: {self.sim_id}")

        # ── 2단계: LLM 아군 기지 선택 ────────────────────────────────────
        self.llm = LLMCommander(
            model_id=llm_model_id,
            device=llm_device,
            db=self.db,
            sim_id=self.sim_id,
        )
        dispatch = self.llm.decide_friendly_dispatch(self.enemy_bases_selected)
        logger.info(f"LLM 출격 결정:\n{dispatch}")

        self.dispatch_plan: List[Dict] = dispatch["friendly_dispatch"]
        # dispatch_plan[i] = {"base": "강릉공군기지", "oppose": "원산기지"}

        # ── 편대쌍 구성 ──────────────────────────────────────────────────
        self.formation_pairs: List[Dict] = []
        used_friendly_bases: List[str] = []

        for i, plan in enumerate(self.dispatch_plan):
            friendly_base_name = plan["base"]
            enemy_base_name    = plan["oppose"]

            friendly_base = {"name": friendly_base_name, **FRIENDLY_BASES[friendly_base_name]}
            enemy_base    = {"name": enemy_base_name,    **ENEMY_BASES[enemy_base_name]}

            used_friendly_bases.append(friendly_base_name)

            # UID: 아군 A<i>100~A<i>200, 적군 B<i>100~B<i>200
            f_uid_a = f"A{i}100"
            f_uid_b = f"A{i}200"
            e_uid_a = f"B{i}100"
            e_uid_b = f"B{i}200"

            # 아군 → 적 방향 헤딩
            hdg_f2e = math.degrees(
                _bearing_rad(
                    friendly_base["lon"], friendly_base["lat"],
                    enemy_base["lon"],    enemy_base["lat"],
                )
            )
            # 적 → 아군 방향 헤딩
            hdg_e2f = math.degrees(
                _bearing_rad(
                    enemy_base["lon"],    enemy_base["lat"],
                    friendly_base["lon"], friendly_base["lat"],
                )
            )

            # 편대 내 2대: 약간 간격 오프셋 (0.02° ≈ 2km)
            f_state_a = _base_to_init_state(friendly_base["lon"],          friendly_base["lat"],          hdg_f2e)
            f_state_b = _base_to_init_state(friendly_base["lon"] + 0.02,   friendly_base["lat"],          hdg_f2e)
            e_state_a = _base_to_init_state(enemy_base["lon"],             enemy_base["lat"],             hdg_e2f)
            e_state_b = _base_to_init_state(enemy_base["lon"] + 0.02,      enemy_base["lat"],             hdg_e2f)

            self.formation_pairs.append({
                "friendly_uids": [f_uid_a, f_uid_b],
                "enemy_uids":    [e_uid_a, e_uid_b],
                "friendly_base": friendly_base,
                "enemy_base":    enemy_base,
                "friendly_init_states": {
                    f_uid_a: f_state_a,
                    f_uid_b: f_state_b,
                },
                "enemy_init_states": {
                    e_uid_a: e_state_a,
                    e_uid_b: e_state_b,
                },
            })

        self.used_friendly_bases = used_friendly_bases

        # ── DB 편대 정보 등록 ─────────────────────────────────────────────
        friendly_db_ids: Dict[int, int] = {}
        enemy_db_ids: Dict[int, int] = {}
        for i, pair in enumerate(self.formation_pairs):
            fid = self.db.add_formation(
                sim_id=self.sim_id,
                team="friendly",
                base_name=pair["friendly_base"]["name"],
                base_lon=pair["friendly_base"]["lon"],
                base_lat=pair["friendly_base"]["lat"],
                aircraft_uids=pair["friendly_uids"],
                paired_enemy_id=None,
            )
            eid = self.db.add_formation(
                sim_id=self.sim_id,
                team="enemy",
                base_name=pair["enemy_base"]["name"],
                base_lon=pair["enemy_base"]["lon"],
                base_lat=pair["enemy_base"]["lat"],
                aircraft_uids=pair["enemy_uids"],
            )
            # paired_enemy_id 업데이트
            self.db.db_path  # noqa: just accessing attribute (no-op)
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            conn.execute(
                "UPDATE formation_info SET paired_enemy_id=? WHERE formation_id=?",
                (eid, fid),
            )
            conn.commit()
            conn.close()
            friendly_db_ids[i] = fid
            enemy_db_ids[i] = eid

        # ── 환경 생성 ─────────────────────────────────────────────────────
        logger.info("TacticalCombatEnv 초기화 중...")
        self.env = TacticalCombatEnv(
            formation_pairs=self.formation_pairs,
            db=self.db,
            sim_id=self.sim_id,
            ego_policy_path=ego_policy_path,
            enm_policy_path=enm_policy_path,
            battle_field_center=self.battle_field_center,
            device=device,
        )
        self.env.register_formation_ids(friendly_db_ids, enemy_db_ids)
        logger.info("환경 초기화 완료.")

        # 이벤트 처리 완료 플래그
        self._major_loss_handled: bool = False
        self._ammo_handled_uids: set = set()
        self._formation_ammo_handled: set = set()        # 처리 완료된 pair_idx
        self._formation_destroyed_handled: set = set()   # 처리 완료된 pair_idx
        self._enemy_formation_destroyed_handled: set = set()  # 처리 완료된 pair_idx

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------

    def run(self):
        """시뮬레이션 메인 루프."""
        logger.info("=" * 60)
        logger.info("전술 시뮬레이션 시작")
        logger.info("=" * 60)

        obs, share_obs = self.env.reset()

        if self.render:
            self.env.render(mode="txt", filepath=self.render_path)

        for step in range(self.max_steps):
            # ── 전술 스텝 실행 ─────────────────────────────────────────
            obs, share_obs, rewards, dones, info = self.env.step_tactical()

            if self.render:
                self.env.render(mode="txt", filepath=self.render_path)

            # ── 이벤트 처리 ────────────────────────────────────────────
            self._handle_events(info, step)

            # ── 종료 조건 ──────────────────────────────────────────────
            if self._is_simulation_done(dones, info):
                logger.info(f"시뮬레이션 종료 (step={step})")
                break

            # 진행 상황 로그 (100스텝마다)
            if step % 100 == 0:
                self._log_status(step)

        self.db.update_simulation_status(self.sim_id, "completed")
        logger.info("시뮬레이션 완료. DB 저장 완료.")

    # ------------------------------------------------------------------
    # 이벤트 처리
    # ------------------------------------------------------------------

    def _handle_events(self, info: Dict, step: int):
        ts = step * self.env.time_interval

        # ── 5.1 아군 전력 50% 이상 손실 ───────────────────────────────
        if info.get("event_major_loss") and not self._major_loss_handled:
            event_id = info["event_major_loss_id"]
            logger.info(f"[이벤트 처리] major_loss (event_id={event_id})")

            decision = self.llm.decide_on_major_loss(
                event_id=event_id,
                step=step,
                timestamp=ts,
            )
            action = decision.get("action", "rtb")
            logger.info(f"LLM 결정: {action} — {decision.get('reasoning','')}")

            if action == "rtb":
                # 6.1.1: 전체 RTB
                self.env.handle_major_loss_rtb()
            else:
                # 6.1.2: 지원 요청 — 교전 중인 첫 번째 페어 기준
                pair_idx = self._active_combat_pair()
                if pair_idx is not None:
                    combat_lon, combat_lat = self.env._formation_centroid(
                        self.formation_pairs[pair_idx]["enemy_uids"]
                    )
                    support_base = _closest_friendly_base(
                        combat_lon, combat_lat,
                        exclude=self.used_friendly_bases,
                    )
                    if support_base is None:
                        # 이미 모든 기지 사용 중 → 기존 기지 중 가장 가까운 곳
                        support_base = _closest_friendly_base(
                            combat_lon, combat_lat
                        )
                    if support_base:
                        logger.info(f"지원 기지: {support_base['name']}")
                        self.env.handle_request_support(support_base, pair_idx)

            self._major_loss_handled = True

        # ── 5.2 무장 고갈 ──────────────────────────────────────────────
        if info.get("event_ammo_depleted"):
            ammo_ids: Dict[str, int] = info.get("event_ammo_depleted_ids", {})
            for uid, event_id in ammo_ids.items():
                if uid in self._ammo_handled_uids:
                    continue
                logger.info(f"[이벤트 처리] ammo_depleted uid={uid} (event_id={event_id})")

                decision = self.llm.decide_on_ammo_depletion(
                    event_id=event_id,
                    aircraft_uid=uid,
                    step=step,
                    timestamp=ts,
                )
                action = decision.get("action", "rtb")
                logger.info(
                    f"LLM 결정 ({uid}): {action} — {decision.get('reasoning','')}"
                )

                if action == "rtb":
                    # 6.2.1: RTB → 재장착 → 복귀
                    self.env.handle_ammo_rtb(uid)

                # continue: 아무 조치 없이 계속 비행
                self._ammo_handled_uids.add(uid)

        # ── 5.3 편대 단위 무장 고갈 ────────────────────────────────────
        formation_ammo: Dict[int, int] = info.get("event_formation_ammo_depleted", {})
        for pair_idx_str, event_id in formation_ammo.items():
            pair_idx = int(pair_idx_str)
            if pair_idx in self._formation_ammo_handled:
                continue
            pair = self.formation_pairs[pair_idx]
            logger.info(
                f"[이벤트 처리] formation_ammo_depleted "
                f"pair={pair_idx} ({pair['friendly_base']['name']}) "
                f"(event_id={event_id})"
            )

            # 교전 지역 중심 (적 편대 위치)
            combat_lon, combat_lat = self.env._formation_centroid(
                pair["enemy_uids"]
            )

            # 근처 미사용 기지에서 지원 편대 출격
            support_base = _closest_friendly_base(
                combat_lon, combat_lat,
                exclude=self.used_friendly_bases,
            )
            if support_base is None:
                # 모든 기지 사용 중 → 가장 가까운 기지 재사용
                support_base = _closest_friendly_base(combat_lon, combat_lat)

            if support_base:
                logger.info(
                    f"[편대 무장고갈] 지원 편대 출격: {support_base['name']} → pair {pair_idx}"
                )
                self.env.handle_request_support(support_base, pair_idx)

            # 무장 고갈된 편대 전체 RTB
            self.env.handle_formation_ammo_rtb(pair_idx)
            self._formation_ammo_handled.add(pair_idx)

        # ── 5.4 아군 편대 전멸 → LLM 교체 편대 출격 ────────────────────
        formation_destroyed: Dict[int, int] = info.get("event_formation_destroyed", {})
        for pair_idx_str, event_id in formation_destroyed.items():
            pair_idx = int(pair_idx_str)
            if pair_idx in self._formation_destroyed_handled:
                continue
            pair = self.formation_pairs[pair_idx]
            logger.info(
                f"[이벤트 처리] formation_destroyed "
                f"pair={pair_idx} ({pair['friendly_base']['name']}) "
                f"(event_id={event_id})"
            )

            # 잔여 적기 수
            alive_enemy_count = sum(
                1 for u in pair["enemy_uids"]
                if u in self.env.agents and self.env.agents[u].is_alive
            )

            # LLM에게 교체 기지 결정 요청
            # 모든 아군 기지를 후보로 제공 (이미 사용 중이어도 재사용 가능)
            available = list(FRIENDLY_BASES.keys())
            decision = self.llm.decide_on_formation_destroyed(
                event_id=event_id,
                pair_idx=pair_idx,
                destroyed_base=pair["friendly_base"]["name"],
                enemy_base=pair["enemy_base"]["name"],
                alive_enemy_count=alive_enemy_count,
                available_bases=available,
                step=step,
                timestamp=ts,
            )
            chosen_base_name = decision.get("base", "")
            logger.info(
                f"LLM 교체 기지 결정: {chosen_base_name} "
                f"— {decision.get('reasoning', '')}"
            )

            if chosen_base_name and chosen_base_name in FRIENDLY_BASES:
                replacement_base = {
                    "name": chosen_base_name,
                    **FRIENDLY_BASES[chosen_base_name],
                }
                self.env.handle_spawn_replacement(replacement_base, pair_idx)
            else:
                logger.warning("교체 기지 결정 실패. 교체 편대 미출격.")

            self._formation_destroyed_handled.add(pair_idx)

        # ── 5.5 적 편대 전멸 → LLM 재배치/RTB 결정 ─────────────────────
        enemy_destroyed: Dict[int, int] = info.get("event_enemy_formation_destroyed", {})
        for pair_idx_str, event_id in enemy_destroyed.items():
            pair_idx = int(pair_idx_str)
            if pair_idx in self._enemy_formation_destroyed_handled:
                continue
            pair = self.formation_pairs[pair_idx]
            logger.info(
                f"[이벤트 처리] enemy_formation_destroyed "
                f"pair={pair_idx} ({pair['friendly_base']['name']} → {pair['enemy_base']['name']}) "
                f"(event_id={event_id})"
            )

            alive_friendly = sum(
                1 for u in pair["friendly_uids"]
                if u in self.env.agents and self.env.agents[u].is_alive
            )

            # 다른 살아있는 적 편대 목록
            other_enemy_formations = []
            for other_idx, other_pair in enumerate(self.formation_pairs):
                if other_idx == pair_idx:
                    continue
                alive_enemies = [
                    u for u in other_pair["enemy_uids"]
                    if u in self.env.agents and self.env.agents[u].is_alive
                ]
                if not alive_enemies:
                    continue
                elon, elat = self.env._formation_centroid(other_pair["enemy_uids"])
                other_enemy_formations.append({
                    "pair_idx":   other_idx,
                    "enemy_base": other_pair["enemy_base"]["name"],
                    "alive_enemy": len(alive_enemies),
                    "lon": elon,
                    "lat": elat,
                })

            decision = self.llm.decide_on_enemy_formation_destroyed(
                event_id=event_id,
                pair_idx=pair_idx,
                friendly_base=pair["friendly_base"]["name"],
                enemy_base=pair["enemy_base"]["name"],
                alive_friendly=alive_friendly,
                other_enemy_formations=other_enemy_formations,
                step=step,
                timestamp=ts,
            )
            action = decision.get("action", "rtb")
            logger.info(
                f"LLM 결정 (pair={pair_idx}): {action} — {decision.get('reasoning', '')}"
            )

            if action == "engage" and other_enemy_formations:
                target_pair_idx = decision.get(
                    "target_pair_idx", other_enemy_formations[0]["pair_idx"]
                )
                self.env.handle_enemy_victory_engage(pair_idx, target_pair_idx)
            else:
                self.env.handle_enemy_victory_rtb(pair_idx)

            self._enemy_formation_destroyed_handled.add(pair_idx)

    # ------------------------------------------------------------------
    # 유틸
    # ------------------------------------------------------------------

    def _active_combat_pair(self) -> Optional[int]:
        """현재 교전 중인 첫 번째 편대쌍 인덱스 반환."""
        for i, phase in self.env.pair_phases.items():
            if phase == "combat":
                return i
        return 0  # fallback

    def _is_simulation_done(self, dones: np.ndarray, info: Dict) -> bool:
        """종료 조건 체크."""
        # 모든 기체 done
        if hasattr(dones, "all") and dones.all():
            return True
        # 모든 편대쌍이 rtb_loss / rtb_victory / done
        all_terminal = all(
            p in ("rtb_loss", "rtb_victory", "done")
            for p in self.env.pair_phases.values()
        )
        if all_terminal:
            # RTB 완료 체크: 잔존 기체들이 기지에 근접했는지
            for i, pair in enumerate(self.formation_pairs):
                if self.env.pair_phases[i] not in ("rtb_loss", "rtb_victory"):
                    continue
                for uid in pair["friendly_uids"]:
                    if uid not in self.env.agents:
                        continue
                    sim = self.env.agents[uid]
                    if not sim.is_alive:
                        continue
                    lon, lat, _ = sim.get_geodetic()
                    base = pair["friendly_base"]
                    dist = _haversine_m(lon, lat, base["lon"], base["lat"])
                    if dist > 5_000:
                        return False   # 아직 귀환 중
            return True
        return False

    def _log_status(self, step: int):
        phases = self.env.pair_phases
        alive_ego = sum(
            1 for uid in self.env.ego_ids
            if uid in self.env.agents and self.env.agents[uid].is_alive
        )
        alive_enm = sum(
            1 for uid in self.env.enm_ids
            if uid in self.env.agents and self.env.agents[uid].is_alive
        )
        logger.info(
            f"[step={step}] 아군 생존={alive_ego} | 적군 생존={alive_enm} | "
            f"단계={phases}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI 엔트리포인트
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="한반도 전술 공중전 시뮬레이터")
    p.add_argument("--ego_policy",  default="",
                   help="아군 PPOActor 체크포인트 경로 (.pt)")
    p.add_argument("--enm_policy",  default="",
                   help="적군 PPOActor 체크포인트 경로 (.pt)")
    p.add_argument("--llm_model",
                   default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
                   help="EXAONE 모델 ID (HuggingFace)")
    p.add_argument("--db_path",     default="combat_simulation.db",
                   help="SQLite DB 파일 경로")
    p.add_argument("--max_steps",   type=int, default=3000,
                   help="시뮬레이션 최대 스텝")
    p.add_argument("--render",      action="store_true",
                   help="TacView ACMI 파일 렌더링")
    p.add_argument("--render_path", default="tactical_combat.txt.acmi",
                   help="렌더 출력 파일 경로")
    p.add_argument("--device",      default="cpu",
                   help="RL 정책 디바이스 (cpu / cuda)")
    p.add_argument("--llm_device",  default="auto",
                   help="LLM 디바이스 (auto / cuda / cpu)")
    p.add_argument("--seed",        type=int, default=42,
                   help="랜덤 시드")
    p.add_argument("--num_enemy",   type=int, default=None,
                   help="적 편대 수 지정 (기본: 랜덤 1~3)")
    return p.parse_args()


def main():
    args = parse_args()
    controller = TacticalController(
        ego_policy_path=args.ego_policy,
        enm_policy_path=args.enm_policy,
        llm_model_id=args.llm_model,
        db_path=args.db_path,
        max_steps=args.max_steps,
        render=args.render,
        render_path=args.render_path,
        device=args.device,
        llm_device=args.llm_device,
        seed=args.seed,
        num_enemy_formations=args.num_enemy,
    )
    controller.run()


if __name__ == "__main__":
    main()
