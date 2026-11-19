"""复核本轮文件指纹并从原始记录重新评分；不训练、不写入证据包。"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from black.checkpoint import source_identity as black_source  # noqa: E402
from black.course_evaluation import score_attempt as course_score  # noqa: E402
from black.eval_score import score_attempt as base_score  # noqa: E402
from wolf.checkpoint import source_identity as wolf_source  # noqa: E402
from wolf.eval_score import score as wolf_score  # noqa: E402


def main():
  evidence = ROOT / "evidence/boundary_review_20261119"
  validation = json.loads((evidence / "VALIDATION.json").read_text())
  identities = {"black": black_source(), "wolf": wolf_source()}
  for robot, identity in identities.items():
    assert identity["sha256"] == validation["source_identity"][robot]["sha256"]
  counts = {}
  for name, base in (("SOURCE_INDEX.json", ROOT), ("EVIDENCE_INDEX.json", evidence)):
    rows = json.loads((evidence / name).read_text())["files"]
    for row in rows:
      content = (base / row["path"]).read_bytes()
      assert len(content) == row["bytes"], row["path"]
      assert hashlib.sha256(content).hexdigest() == row["sha256"], row["path"]
    counts[name] = len(rows)
  raw = json.loads((evidence / "RAW_EVAL_MATRIX.json").read_text())
  for row in raw:
    directory = evidence / row["evidence"]
    if row["robot"] == "black":
      scorer = (
        base_score if row["course"] in ("flat8", "axes", "stairs") else course_score
      )
      score = scorer(directory / "MANIFEST.json", directory / "RESULT.json")
      assert score["counts"]["success"] == 0 and not score["collection_complete"]
    else:
      score = wolf_score(directory)
      assert not score["complete"] and not score["qualified"]
    assert score == json.loads((directory / "SCORE.json").read_text())
  print(
    json.dumps(
      dict(
        source_assets_match=True,
        hashes_match=True,
        indexed_files=counts,
        raw_suites_recomputed=len(raw),
        raw_trials=sum(row["assigned"] for row in raw),
        all_raw_scores_identical=True,
        qualified=False,
        runtime_matches_JC={
          robot: identities[robot]["runtime"]
          == validation["source_identity"][robot]["runtime"]
          for robot in identities
        },
      ),
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
