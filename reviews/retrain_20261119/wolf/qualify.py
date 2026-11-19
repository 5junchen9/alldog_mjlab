"""重读原始attempt建立指定课程的晋级/学生资格。"""

import argparse
import json
from pathlib import Path

from wolf.checkpoint import load_bundle, teacher_identity
from wolf.curriculum import COURSES
from wolf.eval_score import require_qualification
from wolf.run import validate_cap


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--index", type=Path, required=True)
  parser.add_argument("--bundle", type=Path, required=True)
  parser.add_argument("--course", choices=COURSES, required=True)
  parser.add_argument("--student", action="store_true")
  parser.add_argument("--max-level", type=int)
  args = parser.parse_args()
  validate_cap(parser, args.course, args.max_level)
  teacher, _, _, _ = load_bundle(args.bundle)
  stage = "student:" + args.course if args.student else args.course
  print(
    json.dumps(
      require_qualification(
        args.index, teacher_identity(teacher), stage, args.max_level
      ),
      ensure_ascii=False,
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
