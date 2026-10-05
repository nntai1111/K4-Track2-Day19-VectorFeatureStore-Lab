"""Bonus demo — 5 queries through HybridMemoryAgent.  Run: python bonus/demo.py"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import HybridMemoryAgent  # noqa: E402
from profile_store import seed_profiles  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MEMORIES_U001 = [
    ("kubernetes", "doc", "Hôm nay mình đọc bài về Kubernetes. Pod là đơn vị deploy nhỏ nhất, "
     "Deployment quản lý ReplicaSet để rolling update không downtime. "
     "Cần nhớ: readiness probe khác liveness probe."),
    ("cloud", "doc", "Điện toán đám mây cho phép autoscaling: hệ thống tự thêm máy khi lưu lượng "
     "tăng và bớt máy khi rảnh. HPA trong Kubernetes scale theo CPU hoặc custom metric."),
    ("cloud", "note", "Ghi chú: để giảm chi phí cloud nên dùng spot instance cho batch job "
     "và reserved instance cho workload ổn định."),
    ("security", "doc", "Cloud security: nguyên tắc least privilege với IAM, bật MFA cho tài khoản root, "
     "mã hoá dữ liệu at rest bằng KMS và rotate key định kỳ."),
    ("security", "chat", "Mình hỏi trợ lý về Nghị định 13/2023 bảo vệ dữ liệu cá nhân: "
     "dữ liệu nhạy cảm cần sự đồng ý rõ ràng của chủ thể dữ liệu."),
    ("ai_ml", "doc", "RAG pipeline: chunk tài liệu, embed, lưu vector store, retrieve top-k rồi "
     "đưa vào prompt. Hybrid search với RRF giúp bắt cả keyword lẫn ngữ nghĩa."),
    ("personal", "note", "Nhắc việc: cuối tuần đi cafe với team, review lại slide thuyết trình."),
]

QUERIES = [
    ("1. simple vector hit", "Tôi đã đọc gì về Kubernetes?"),
    ("2. needs profile", "Recommend đọc gì tiếp"),
    ("3. needs fresh activity", "Tôi đang quan tâm gì gần đây?"),
    ("4. paraphrase", "Tài liệu về tự động mở rộng hạ tầng?"),
    ("5. mixed hybrid + profile", "Cho tôi summary cloud security"),
]


def main() -> None:
    agent = HybridMemoryAgent()
    seed_profiles(agent.fs, [
        {"user_id": "u_001", "preferred_language": "mix", "reading_speed_wpm": 220,
         "topic_affinity": "cloud", "active_hours": "20-23"},
        {"user_id": "u_002", "preferred_language": "vi", "reading_speed_wpm": 180,
         "topic_affinity": "finance", "active_hours": "8-11"},
    ])
    n = sum(agent.remember(text, "u_001", topic=t, source=s) for t, s, text in MEMORIES_U001)
    agent.remember("Bí mật: doanh thu quý 3 của công ty là 4,2 tỷ VND, chưa công bố.",
                   "u_002", topic="finance")
    print(f"Indexed {n} memory chunks for u_001 (+1 private chunk for u_002)\n")

    for label, q in QUERIES:
        print(f"=== {label}: {q!r}")
        print(agent.recall(q, user_id="u_001"))
        print()

    # Isolation check: u_001 must never see u_002's memory, even on a direct keyword hit.
    leak = agent.recall("doanh thu quý 3", user_id="u_001")
    assert "4,2 tỷ" not in leak, "cross-user leak!"
    print("=== isolation: u_001 asking 'doanh thu quý 3' does NOT see u_002's memory  -> OK")


if __name__ == "__main__":
    main()
