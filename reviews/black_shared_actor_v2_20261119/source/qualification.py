"""重新评分原始attempt的阶段门；手填布尔值和历史SCORE不能授予资格。"""

import argparse
import json
from pathlib import Path

from eval_protocol import EVALUATION_SEEDS, QUALIFICATION_ID, roster
from eval_score import file_sha256, require, score_attempt

REQUIRED_SUITES = {
  "flat_axes": ("flat8",),
  "stairs": ("flat8", "axes"),
  "student": ("flat8", "axes", "stairs"),
}


def require_qualification(path, teacher_id, stage, *, source_sha256) -> dict:
  """index只给原始文件位置/hash，所有通过结论在这里重算。"""
  require(stage in REQUIRED_SUITES, "unknown qualification stage")
  path = Path(path)
  index = json.loads(path.read_text())
  require(
    index.get("schema") == QUALIFICATION_ID, "qualification index schema mismatch"
  )
  expected = {
    (suite, seed) for suite in REQUIRED_SUITES[stage] for seed in EVALUATION_SEEDS
  }
  require(len(index["attempts"]) == len(expected), "incomplete fixed multi-seed panel")
  reports, seen, bundles = [], set(), set()
  for entry in index["attempts"]:
    directory = Path(entry["path"])
    if not directory.is_absolute():
      directory = path.parent / directory
    manifest_path, result_path = directory / "MANIFEST.json", directory / "RESULT.json"
    require(
      file_sha256(manifest_path) == entry["manifest_sha256"], "indexed manifest changed"
    )
    require(
      file_sha256(result_path) == entry["result_sha256"], "indexed result changed"
    )
    manifest = json.loads(manifest_path.read_text())
    key = (manifest["suite"], manifest["base_seed"])
    require(key in expected and key not in seen, "unknown or repeated panel cell")
    require(manifest["roster"] == roster(*key), "fixed roster coverage mismatch")
    require(
      manifest["max_controls"] is None,
      "administratively limited attempt cannot qualify",
    )
    require(
      manifest["teacher_id"] == teacher_id, "qualification belongs to another teacher"
    )
    require(
      manifest["source_sha256"] == source_sha256,
      "qualification source differs from current source",
    )
    result = json.loads(result_path.read_text())
    require(result["collection_complete"] is True, "incomplete collection")
    report = score_attempt(manifest_path, result_path)
    counts = report["counts"]
    require(report["collection_complete"], "incomplete raw evidence")
    require(
      counts["success"] == counts["assigned"],
      "failed or invalid trial blocks qualification",
    )
    require(
      counts["independent_episodes"] == counts["assigned"],
      "independent episode coverage incomplete",
    )
    require(
      all(item["invalid_depth_frames"] == 0 for item in report["trials"]),
      "invalid depth blocks qualification",
    )
    bundles.add(report["bundle_sha256"])
    seen.add(key)
    reports.append(
      dict(
        suite=key[0],
        seed=key[1],
        assigned=counts["assigned"],
        manifest_sha256=report["manifest_sha256"],
        result_sha256=report["result_sha256"],
      )
    )
  require(seen == expected and len(bundles) == 1, "panel identity or coverage mismatch")
  return dict(
    schema=QUALIFICATION_ID,
    stage=stage,
    teacher_id=teacher_id,
    source_sha256=source_sha256,
    passed=True,
    recomputed=True,
    attempts=reports,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", choices=tuple(REQUIRED_SUITES), required=True)
  parser.add_argument("--attempts", type=Path, nargs="+", required=True)
  parser.add_argument("--teacher-id", required=True)
  parser.add_argument("--source-sha256", required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  entries = [
    dict(
      path=str(directory.resolve()),
      manifest_sha256=file_sha256(directory / "MANIFEST.json"),
      result_sha256=file_sha256(directory / "RESULT.json"),
    )
    for directory in args.attempts
  ]
  with args.out.open("x") as stream:
    json.dump(dict(schema=QUALIFICATION_ID, attempts=entries), stream, indent=2)
  # 即使该index存在，入口仍每次重算；未通过的index永远不能放行。
  print(
    json.dumps(
      require_qualification(
        args.out, args.teacher_id, args.stage, source_sha256=args.source_sha256
      ),
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
