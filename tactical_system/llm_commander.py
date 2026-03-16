"""
llm_commander.py
================
EXAONE-3.5 기반 LLM 전술 지휘관.

역할:
  1. 아군 출격 기지 선택 (적 편대 수 기반)
  2. 각 아군 편대와 대응할 적 편대 매칭
  3. 이벤트 발생 시 RTB 여부 / 지원 요청 여부 판단
     - 50% 이상 손실: 전체 RTB 또는 지원 요청
     - 무장 고갈: 개별 RTB 여부 판단

사용 모델: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct (HuggingFace transformers)
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .combat_db import CombatDB

logger = logging.getLogger(__name__)


# ── 한반도 기지 정보 ─────────────────────────────────────────────────────────

ENEMY_BASES = {
    "원산기지": {
        "lon": 127.41, "lat": 39.17,
        "description": "원산 공군 기지 (북한 동해안)",
    },
    "평양공군기지": {
        "lon": 125.67, "lat": 39.20,
        "description": "평양 순안 공군 기지 (북한 수도권)",
    },
    "순천공군기지": {
        "lon": 125.73, "lat": 39.41,
        "description": "순천 공군 기지 (북한 서부)",
    },
}

FRIENDLY_BASES = {
    "성남공군기지": {
        "lon": 127.12, "lat": 37.45,
        "description": "성남 공군 기지 / K-16 (수도권)",
    },
    "강릉공군기지": {
        "lon": 128.95, "lat": 37.75,
        "description": "강릉 공군 기지 (동해안)",
    },
    "청주공군기지": {
        "lon": 127.49, "lat": 36.72,
        "description": "청주 공군 기지 (중부)",
    },
}


class LLMCommander:
    """EXAONE-3.5 기반 전술 AI 지휘관."""

    def __init__(
        self,
        model_id: str = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        device: str = "auto",
        db: Optional[CombatDB] = None,
        sim_id: int = -1,
        max_new_tokens: int = 512,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.db = db
        self.sim_id = sim_id

        logger.info(f"EXAONE-3.5 모델 로딩 중: {model_id}")
        # local_files_only=True: 패쇄망 환경에서 HuggingFace Hub 접근 시도 차단
        # 모델은 ~/.cache/huggingface/ 에 사전 캐싱되어 있어야 함
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        logger.info("모델 로딩 완료.")

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        """EXAONE chat template을 사용해 텍스트 생성."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # decode only the newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """응답 텍스트에서 JSON 블록 추출."""
        # ```json ... ``` 블록 우선
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # 중괄호 패턴 fallback
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    # ------------------------------------------------------------------
    # 1 & 2단계: 아군 출격 기지 선택 및 적 편대 대응 매칭
    # ------------------------------------------------------------------

    def decide_friendly_dispatch(
        self,
        enemy_bases_selected: List[str],
    ) -> Dict:
        """
        아군 출격 기지 및 각 편대의 대응 적 편대를 LLM이 결정.

        Parameters
        ----------
        enemy_bases_selected : List[str]
            랜덤 선택된 적 출격 기지 이름 목록

        Returns
        -------
        dict  예시:
          {
            "friendly_dispatch": [
              {"base": "강릉공군기지", "oppose": "원산기지"},
              {"base": "청주공군기지", "oppose": "평양공군기지"}
            ],
            "reasoning": "..."
          }
        """
        n = len(enemy_bases_selected)
        available_friendly = list(FRIENDLY_BASES.keys())

        system_prompt = (
            "당신은 대한민국 공군 전술 지휘관 AI입니다. "
            "주어진 적 편대 출격 정보를 바탕으로 아군 출격 기지를 선택하고 "
            "각 아군 편대가 어떤 적 편대에 대응할지 지시하십시오. "
            "반드시 JSON 형식으로만 응답하십시오."
        )

        enemy_info = "\n".join(
            f"  - {b}: {ENEMY_BASES[b]['description']} "
            f"(위치: 경도 {ENEMY_BASES[b]['lon']:.2f}°, 위도 {ENEMY_BASES[b]['lat']:.2f}°)"
            for b in enemy_bases_selected
        )
        friendly_info = "\n".join(
            f"  - {b}: {FRIENDLY_BASES[b]['description']} "
            f"(위치: 경도 {FRIENDLY_BASES[b]['lon']:.2f}°, 위도 {FRIENDLY_BASES[b]['lat']:.2f}°)"
            for b in available_friendly
        )

        user_prompt = f"""
현재 상황:
- 적 편대 출격 기지 ({n}개):
{enemy_info}

- 가용 아군 기지 (총 3개 중 {n}개 선택):
{friendly_info}

임무:
1. 아군 기지 {n}개를 선택하여 각각 1개 편대씩 출격시키십시오.
2. 각 아군 편대가 어떤 적 편대에 대응할지 1:1로 매칭하십시오.
3. 지리적 위치, 대응 효율성, 방어 우선순위를 고려하여 판단하십시오.

다음 JSON 형식으로 응답하십시오:
```json
{{
  "friendly_dispatch": [
    {{"base": "아군기지명", "oppose": "대응할_적기지명"}},
    ...
  ],
  "reasoning": "판단 근거"
}}
```
"""
        raw = self._generate(system_prompt, user_prompt)
        logger.debug(f"[LLM dispatch response]\n{raw}")

        result = self._extract_json(raw)
        if result is None:
            logger.warning("LLM 응답 JSON 파싱 실패. 기본값 사용.")
            # fallback: 순서대로 매칭
            result = {
                "friendly_dispatch": [
                    {"base": available_friendly[i % len(available_friendly)],
                     "oppose": enemy_bases_selected[i]}
                    for i in range(n)
                ],
                "reasoning": "파싱 실패 — 기본 순서 매칭",
            }

        # DB 기록
        if self.db and self.sim_id >= 0:
            self.db.log_llm_decision(
                sim_id=self.sim_id,
                step=0,
                timestamp=0.0,
                decision_type="dispatch",
                input_prompt=user_prompt,
                output_decision=json.dumps(result, ensure_ascii=False),
                reasoning=result.get("reasoning", ""),
            )
        return result

    # ------------------------------------------------------------------
    # 3단계: 아군-적군 편대 1:1 배정 (enemies 리스트 설정 전 LLM 판단)
    # ------------------------------------------------------------------

    def assign_formation_targets(self, formation_pairs: List[Dict]) -> List[Dict]:
        """
        아군 편대 각각이 어느 적 편대를 전담 교전할지 LLM이 결정.

        Parameters
        ----------
        formation_pairs : list of dict
            TacticalCombatEnv.formation_pairs 구조:
            [{"friendly_base": {"name":…,"lon":…,"lat":…},
              "enemy_base":    {"name":…,"lon":…,"lat":…},
              "friendly_uids": […], "enemy_uids": […]}, …]

        Returns
        -------
        list of dict  예시:
          [
            {"friendly_base": "성남공군기지", "enemy_base": "원산기지"},
            {"friendly_base": "강릉공군기지", "enemy_base": "평양공군기지"}
          ]
          TacticalCombatEnv.apply_formation_assignment() 에 직접 전달 가능.
        """
        n = len(formation_pairs)

        # 편대 정보 문자열 생성
        friendly_lines = []
        enemy_lines = []
        for i, pair in enumerate(formation_pairs):
            fb = pair["friendly_base"]
            eb = pair["enemy_base"]
            f_info = FRIENDLY_BASES.get(fb["name"], {})
            e_info = ENEMY_BASES.get(eb["name"], {})
            f_desc = f_info.get("description", fb["name"])
            e_desc = e_info.get("description", eb["name"])
            friendly_lines.append(
                f"  - {fb['name']}: {f_desc} "
                f"(경도 {fb['lon']:.2f}°, 위도 {fb['lat']:.2f}°, "
                f"출격 기체 {len(pair['friendly_uids'])}대)"
            )
            enemy_lines.append(
                f"  - {eb['name']}: {e_desc} "
                f"(경도 {eb['lon']:.2f}°, 위도 {eb['lat']:.2f}°, "
                f"출격 기체 {len(pair['enemy_uids'])}대)"
            )

        system_prompt = (
            "당신은 대한민국 공군 전술 지휘관 AI입니다. "
            "아군 편대와 적 편대를 1:1로 배정하여 각 아군 편대가 "
            "어느 적 편대를 전담 교전할지 결정하십시오. "
            "지리적 근접성, 방어 우선순위, 전술적 효율성을 고려하십시오. "
            "반드시 JSON 형식으로만 응답하십시오."
        )

        user_prompt = f"""
편대 배정 임무:
- 아군 편대 ({n}개):
{chr(10).join(friendly_lines)}

- 적군 편대 ({n}개):
{chr(10).join(enemy_lines)}

각 아군 편대가 전담할 적 편대를 1:1로 배정하십시오.
지리적으로 가까운 기지끼리 배정하되, 전술적 균형도 고려하십시오.

다음 JSON 형식으로 응답하십시오:
```json
{{
  "assignment": [
    {{"friendly_base": "아군기지명", "enemy_base": "담당_적기지명"}},
    ...
  ],
  "reasoning": "배정 근거"
}}
```
"""
        raw = self._generate(system_prompt, user_prompt)
        logger.debug(f"[LLM assign_formation_targets response]\n{raw}")

        result = self._extract_json(raw)
        assignment = None
        if result and "assignment" in result:
            assignment = result["assignment"]

        if not assignment or len(assignment) != n:
            logger.warning("편대 배정 파싱 실패 또는 수 불일치. 기본값(순서 매칭) 사용.")
            assignment = [
                {
                    "friendly_base": formation_pairs[i]["friendly_base"]["name"],
                    "enemy_base":    formation_pairs[i]["enemy_base"]["name"],
                }
                for i in range(n)
            ]

        # DB 기록
        if self.db and self.sim_id >= 0:
            self.db.log_llm_decision(
                sim_id=self.sim_id,
                step=0,
                timestamp=0.0,
                decision_type="formation_assignment",
                input_prompt=user_prompt,
                output_decision=json.dumps(
                    {"assignment": assignment}, ensure_ascii=False
                ),
                reasoning=(result or {}).get("reasoning", ""),
            )

        return assignment

    # ------------------------------------------------------------------
    # 4단계: 아군 편대 전멸 → 교체 편대 출격 기지 결정
    # ------------------------------------------------------------------

    def decide_on_formation_destroyed(
        self,
        event_id: int,
        pair_idx: int,
        destroyed_base: str,
        enemy_base: str,
        alive_enemy_count: int,
        available_bases: List[str],
        step: int,
        timestamp: float,
    ) -> Dict:
        """
        아군 편대 전멸 시 교체 편대 출격 기지를 LLM이 결정.

        Parameters
        ----------
        event_id          : DB 이벤트 ID
        pair_idx          : 전멸된 편대쌍 인덱스
        destroyed_base    : 전멸된 편대의 원래 기지명
        enemy_base        : 교전 중인 적 기지명
        alive_enemy_count : 남은 적기 수
        available_bases   : 출격 가능한 아군 기지 목록
        step, timestamp   : 현재 시뮬레이션 스텝 / 시각

        Returns
        -------
        dict  예시:
          {
            "base": "청주공군기지",
            "reasoning": "…"
          }
        """
        if not available_bases:
            return {"base": "", "reasoning": "가용 기지 없음"}

        base_info_lines = "\n".join(
            f"  - {b}: {FRIENDLY_BASES.get(b, {}).get('description', b)} "
            f"(경도 {FRIENDLY_BASES.get(b, {}).get('lon', 0):.2f}°, "
            f"위도 {FRIENDLY_BASES.get(b, {}).get('lat', 0):.2f}°)"
            for b in available_bases
        )
        e_info = ENEMY_BASES.get(enemy_base, {})
        e_desc = e_info.get("description", enemy_base)
        e_lon  = e_info.get("lon", 0)
        e_lat  = e_info.get("lat", 0)

        system_prompt = (
            "당신은 대한민국 공군 전술 지휘관 AI입니다. "
            "아군 편대가 전멸한 상황에서 교체 편대 출격 기지를 선택하십시오. "
            "반드시 JSON 형식으로만 응답하십시오."
        )
        user_prompt = f"""
긴급 상황 보고:
- 시뮬레이션 스텝: {step}
- 전멸 편대 기지: {destroyed_base}
- 교전 중인 적 기지: {enemy_base} ({e_desc}, 경도 {e_lon:.2f}°, 위도 {e_lat:.2f}°)
- 잔여 적기 수: {alive_enemy_count}대

아군 편대가 전멸하였습니다. 즉시 교체 편대를 출격시켜야 합니다.
출격 가능한 아군 기지 목록:
{base_info_lines}

위 기지 중 하나를 선택하여 교체 편대를 출격시키십시오.
지리적 근접성(적 기지까지 거리)과 전술적 효율성을 고려하십시오.

다음 JSON 형식으로 응답하십시오:
```json
{{
  "base": "선택한_아군기지명",
  "reasoning": "선택 근거"
}}
```
"""
        raw = self._generate(system_prompt, user_prompt)
        logger.debug(f"[LLM formation_destroyed response]\n{raw}")

        result = self._extract_json(raw)
        chosen_base = (result or {}).get("base", "")

        # 유효성 검증: 가용 기지 목록에 없으면 가장 가까운 기지 선택
        if chosen_base not in available_bases:
            logger.warning(
                f"LLM 선택 기지 '{chosen_base}'가 가용 목록에 없음. "
                f"첫 번째 기지 '{available_bases[0]}' 사용."
            )
            chosen_base = available_bases[0]
            if result is None:
                result = {}
            result["base"] = chosen_base

        if result is None:
            result = {"base": chosen_base, "reasoning": "파싱 실패 — 기본값"}

        if self.db and self.sim_id >= 0:
            self.db.update_event_decision(event_id, result)
            self.db.log_llm_decision(
                sim_id=self.sim_id,
                step=step,
                timestamp=timestamp,
                decision_type="formation_destroyed",
                input_prompt=user_prompt,
                output_decision=json.dumps(result, ensure_ascii=False),
                reasoning=result.get("reasoning", ""),
            )
        return result

    # ------------------------------------------------------------------
    # 5단계: 이벤트 발생 → LLM 판단
    # ------------------------------------------------------------------

    def decide_on_major_loss(
        self,
        event_id: int,
        step: int,
        timestamp: float,
    ) -> Dict:
        """
        아군 전력 50% 이상 손실 이벤트 처리.

        Returns
        -------
        dict  예시:
          {
            "action": "rtb" | "request_support",
            "reasoning": "..."
          }
        """
        if self.db is None:
            return {"action": "rtb", "reasoning": "DB 없음 — 기본 RTB"}

        event = self.db.get_event_by_id(event_id)
        latest_states = self.db.get_latest_aircraft_states(self.sim_id)

        alive_friendly = [
            s for s in latest_states
            if s["team"] == "friendly" and s["is_alive"] == 1
        ]
        dead_friendly = [
            s for s in latest_states
            if s["team"] == "friendly" and s["is_alive"] == 0
        ]
        alive_enemy = [
            s for s in latest_states
            if s["team"] == "enemy" and s["is_alive"] == 1
        ]

        system_prompt = (
            "당신은 대한민국 공군 전술 지휘관 AI입니다. "
            "아군이 심각한 피해를 입은 상황에서 최선의 전술적 판단을 내리십시오. "
            "반드시 JSON 형식으로만 응답하십시오."
        )

        user_prompt = f"""
긴급 상황 보고:
- 시뮬레이션 스텝: {step}
- 아군 생존 기체: {len(alive_friendly)}대
- 아군 손실 기체: {len(dead_friendly)}대
- 적 생존 기체: {len(alive_enemy)}대
- 이벤트 세부정보: {json.dumps(event['details_json'], ensure_ascii=False, indent=2)}

생존 아군 기체 현황:
{json.dumps([{k: v for k, v in s.items() if k in ['aircraft_uid','base_name','missiles_left','alt','speed_mps','flight_phase']} for s in alive_friendly], ensure_ascii=False, indent=2)}

판단 요청:
아군 전력이 50% 이상 손실되었습니다.
두 가지 선택지 중 하나를 선택하십시오:
  (A) "rtb" — 잔존 아군 전체를 즉시 기지로 복귀
  (B) "request_support" — 가장 가까운 기지에서 추가 편대 지원 요청

다음 JSON 형식으로 응답하십시오:
```json
{{
  "action": "rtb" 또는 "request_support",
  "reasoning": "판단 근거"
}}
```
"""
        raw = self._generate(system_prompt, user_prompt)
        logger.debug(f"[LLM major_loss response]\n{raw}")

        result = self._extract_json(raw)
        if result is None or result.get("action") not in ("rtb", "request_support"):
            logger.warning("파싱 실패 또는 invalid action. 기본값 rtb 사용.")
            result = {"action": "rtb", "reasoning": "파싱 실패 — 기본 RTB"}

        # DB 반영
        self.db.update_event_decision(event_id, result)
        self.db.log_llm_decision(
            sim_id=self.sim_id,
            step=step,
            timestamp=timestamp,
            decision_type="major_loss",
            input_prompt=user_prompt,
            output_decision=json.dumps(result, ensure_ascii=False),
            reasoning=result.get("reasoning", ""),
        )
        return result

    def decide_on_enemy_formation_destroyed(
        self,
        event_id: int,
        pair_idx: int,
        friendly_base: str,
        enemy_base: str,
        alive_friendly: int,
        other_enemy_formations: List[Dict],
        step: int,
        timestamp: float,
    ) -> Dict:
        """
        적 편대 전멸 시 아군 편대를 인근 적 편대에 재투입할지 기지로 복귀시킬지 결정.

        Parameters
        ----------
        event_id               : DB 이벤트 ID
        pair_idx               : 승리한 아군 편대쌍 인덱스
        friendly_base          : 아군 편대 기지명
        enemy_base             : 방금 전멸시킨 적 기지명
        alive_friendly         : 생존 아군 기체 수
        other_enemy_formations : 다른 살아있는 적 편대 목록
                                 [{"pair_idx": int, "enemy_base": str,
                                   "alive_enemy": int, "lon": float, "lat": float}, …]
        step, timestamp        : 현재 시뮬레이션 스텝 / 시각

        Returns
        -------
        dict  예시:
          {
            "action": "engage" | "rtb",
            "target_pair_idx": <int>,   # "engage" 선택 시에만 포함
            "reasoning": "..."
          }
        """
        if not other_enemy_formations:
            result = {
                "action": "rtb",
                "reasoning": "교전 가능한 다른 적 편대 없음 — 기지 복귀",
            }
            if self.db and self.sim_id >= 0:
                self.db.update_event_decision(event_id, result)
                self.db.log_llm_decision(
                    sim_id=self.sim_id,
                    step=step,
                    timestamp=timestamp,
                    decision_type="enemy_formation_destroyed",
                    input_prompt="(no other enemies)",
                    output_decision=json.dumps(result, ensure_ascii=False),
                    reasoning=result["reasoning"],
                )
            return result

        f_info = FRIENDLY_BASES.get(friendly_base, {})
        enemy_lines = "\n".join(
            f"  - 편대{e['pair_idx']}: {e['enemy_base']} "
            f"(잔여 {e['alive_enemy']}대, 경도 {e['lon']:.2f}°, 위도 {e['lat']:.2f}°)"
            for e in other_enemy_formations
        )

        system_prompt = (
            "당신은 대한민국 공군 전술 지휘관 AI입니다. "
            "아군 편대가 담당 적 편대를 전멸시킨 상황에서 "
            "다음 행동을 결정하십시오. "
            "반드시 JSON 형식으로만 응답하십시오."
        )
        user_prompt = f"""
전투 승리 보고:
- 시뮬레이션 스텝: {step}
- 승리 편대 기지: {friendly_base}
  (위치: 경도 {f_info.get('lon', 0):.2f}°, 위도 {f_info.get('lat', 0):.2f}°)
- 전멸시킨 적 기지: {enemy_base}
- 현재 생존 아군 기체: {alive_friendly}대

현재 교전 중인 다른 적 편대:
{enemy_lines}

판단 요청:
아군 편대가 적 편대를 전멸시켰습니다. 두 가지 선택지 중 하나를 선택하십시오:
  (A) "engage" — 인근 적 편대 중 하나를 선택하여 즉시 재교전
  (B) "rtb"    — 기지로 복귀 (연료·피로도 고려)

"engage" 선택 시 반드시 target_pair_idx(위 목록의 편대 번호)를 함께 지정하십시오.
전술적 효율성(거리·잔여 적 수)과 아군 피로도를 종합 판단하십시오.

다음 JSON 형식으로 응답하십시오:
```json
{{
  "action": "engage" 또는 "rtb",
  "target_pair_idx": <정수, engage 선택 시>,
  "reasoning": "판단 근거"
}}
```
"""
        raw = self._generate(system_prompt, user_prompt)
        logger.debug(f"[LLM enemy_formation_destroyed response]\n{raw}")

        result = self._extract_json(raw)
        valid_targets = {e["pair_idx"] for e in other_enemy_formations}

        if result is None or result.get("action") not in ("engage", "rtb"):
            logger.warning("파싱 실패 또는 invalid action. 기본값 engage(최근접) 사용.")
            result = {
                "action": "engage",
                "target_pair_idx": other_enemy_formations[0]["pair_idx"],
                "reasoning": "파싱 실패 — 최근접 적 편대 재교전",
            }

        if result["action"] == "engage":
            target = result.get("target_pair_idx")
            if target not in valid_targets:
                target = other_enemy_formations[0]["pair_idx"]
                result["target_pair_idx"] = target
                logger.warning(
                    f"target_pair_idx 유효하지 않음. {target}으로 대체."
                )

        if self.db and self.sim_id >= 0:
            self.db.update_event_decision(event_id, result)
            self.db.log_llm_decision(
                sim_id=self.sim_id,
                step=step,
                timestamp=timestamp,
                decision_type="enemy_formation_destroyed",
                input_prompt=user_prompt,
                output_decision=json.dumps(result, ensure_ascii=False),
                reasoning=result.get("reasoning", ""),
            )
        return result

    def decide_on_ammo_depletion(
        self,
        event_id: int,
        aircraft_uid: str,
        step: int,
        timestamp: float,
    ) -> Dict:
        """
        무장 고갈 기체에 대한 RTB 여부 판단.

        Returns
        -------
        dict  예시:
          {
            "action": "rtb" | "continue",
            "aircraft_uid": "A0100",
            "reasoning": "..."
          }
        """
        if self.db is None:
            return {"action": "rtb", "aircraft_uid": aircraft_uid,
                    "reasoning": "DB 없음 — 기본 RTB"}

        event = self.db.get_event_by_id(event_id)
        latest_states = self.db.get_latest_aircraft_states(self.sim_id)

        aircraft_state = next(
            (s for s in latest_states if s["aircraft_uid"] == aircraft_uid), {}
        )
        alive_friendly = [
            s for s in latest_states
            if s["team"] == "friendly" and s["is_alive"] == 1
        ]

        system_prompt = (
            "당신은 대한민국 공군 전술 지휘관 AI입니다. "
            "무장이 고갈된 아군 기체의 처리 방침을 결정하십시오. "
            "반드시 JSON 형식으로만 응답하십시오."
        )

        user_prompt = f"""
무장 고갈 보고:
- 기체 UID: {aircraft_uid}
- 출격 기지: {aircraft_state.get('base_name', '알 수 없음')}
- 현재 고도: {aircraft_state.get('alt', 0):.0f}m
- 현재 속도: {aircraft_state.get('speed_mps', 0):.0f}m/s
- 잔여 미사일: {aircraft_state.get('missiles_left', 0)}발
- 아군 총 생존 기체 수: {len(alive_friendly)}대
- 이벤트 세부정보: {json.dumps(event['details_json'], ensure_ascii=False, indent=2)}

판단 요청:
해당 기체의 무장이 고갈되었습니다.
두 가지 선택지 중 하나를 선택하십시오:
  (A) "rtb" — 출격 기지로 복귀하여 AIM-9L 5발 재장착 후 교전 지역 복귀
  (B) "continue" — 현재 위치에서 공중 지원(wing-man) 역할로 계속 임무 수행

다음 JSON 형식으로 응답하십시오:
```json
{{
  "action": "rtb" 또는 "continue",
  "aircraft_uid": "{aircraft_uid}",
  "reasoning": "판단 근거"
}}
```
"""
        raw = self._generate(system_prompt, user_prompt)
        logger.debug(f"[LLM ammo_depletion response]\n{raw}")

        result = self._extract_json(raw)
        if result is None or result.get("action") not in ("rtb", "continue"):
            logger.warning("파싱 실패 또는 invalid action. 기본값 rtb 사용.")
            result = {"action": "rtb", "aircraft_uid": aircraft_uid,
                      "reasoning": "파싱 실패 — 기본 RTB"}

        result["aircraft_uid"] = aircraft_uid  # 확인용 덮어쓰기

        self.db.update_event_decision(event_id, result)
        self.db.log_llm_decision(
            sim_id=self.sim_id,
            step=step,
            timestamp=timestamp,
            decision_type="ammo_depleted",
            input_prompt=user_prompt,
            output_decision=json.dumps(result, ensure_ascii=False),
            reasoning=result.get("reasoning", ""),
        )
        return result
