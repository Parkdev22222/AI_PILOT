"""
run_with_ui.py
==============
TacticalController + Gradio 대시보드 통합 실행 스크립트.

실행 예시:
  python -m tactical_system.run_with_ui \
      --ego_policy  path/to/ego_actor.pt \
      --enm_policy  path/to/enm_actor.pt \
      --llm_model   LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
      --db_path     combat.db \
      --num_enemy   2 \
      --port        7860 \
      --share

동작 순서:
  1. TacticalController 초기화 (LLM 로드, 적/아군 기지 결정, 환경 생성)
  2. Gradio 대시보드 실행 (DB 폴링으로 실시간 시각화)
  3. UI 우측 하단 '▶ 시뮬레이션 시작' 버튼 클릭 시 시뮬레이션 스레드 시작
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="한반도 전술 공중전 시뮬레이터 + Gradio UI"
    )
    # ── 시뮬레이터 설정 ────────────────────────────────────────────────
    p.add_argument("--ego_policy",  default="",
                   help="아군 PPOActor 체크포인트 (.pt)")
    p.add_argument("--enm_policy",  default="",
                   help="적군 PPOActor 체크포인트 (.pt)")
    p.add_argument("--llm_model",
                   default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
                   help="HuggingFace 모델 ID")
    p.add_argument("--db_path",     default="combat_simulation.db",
                   help="SQLite DB 파일 경로")
    p.add_argument("--max_steps",   type=int, default=3000,
                   help="시뮬레이션 최대 스텝")
    p.add_argument("--render",      action="store_true",
                   help="ACMI 파일 렌더 출력")
    p.add_argument("--render_path", default="tactical_combat.txt.acmi",
                   help="렌더 출력 파일 경로")
    p.add_argument("--device",      default="cpu",
                   help="RL 정책 디바이스 (cpu / cuda)")
    p.add_argument("--llm_device",  default="auto",
                   help="LLM 디바이스 (auto / cuda / cpu)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_enemy",   type=int, default=None,
                   help="적 편대 수 (기본: 랜덤 1~3)")
    # ── Gradio 설정 ────────────────────────────────────────────────────
    p.add_argument("--port",        type=int, default=7860,
                   help="Gradio 서버 포트")
    p.add_argument("--share",       action="store_true",
                   help="Gradio public share 링크 생성")
    p.add_argument("--refresh",     type=float, default=2.0,
                   help="대시보드 자동 갱신 주기 (초)")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # ── 지연 임포트 (경로 설정이 된 이후) ─────────────────────────────
    from tactical_system.tactical_controller import TacticalController
    from tactical_system.gradio_dashboard import TacticalDashboard

    # ── 1. 컨트롤러 초기화 (LLM 로드 포함 — 수십 초 소요 가능) ──────────
    print("=" * 60)
    print("전술 시뮬레이터 초기화 중 (LLM 로드 포함)...")
    print("=" * 60)

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

    print(f"\n[초기화 완료]")
    print(f"  시뮬레이션 ID : {controller.sim_id}")
    print(f"  적군 기지     : {controller.enemy_bases_selected}")
    print(f"  편대쌍 수     : {len(controller.formation_pairs)}")
    print(f"  DB 경로       : {args.db_path}")
    print()

    # ── 2. 시뮬레이션 시작 콜백 (UI 버튼 클릭 시 호출) ──────────────
    def _start_simulation():
        try:
            controller.run()
        except Exception as exc:
            logger.error(f"시뮬레이션 오류: {exc}", exc_info=True)

    # ── 3. Gradio 대시보드 실행 ───────────────────────────────────────
    print(f"\nGradio 대시보드 시작: http://0.0.0.0:{args.port}")
    print("UI 우측 하단 '▶ 시뮬레이션 시작' 버튼을 눌러 시뮬레이션을 시작하세요.")
    print("Ctrl+C 로 종료\n")

    dashboard = TacticalDashboard(
        db_path=args.db_path,
        sim_id=controller.sim_id,
        refresh_interval=args.refresh,
        start_callback=_start_simulation,
    )
    dashboard.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
