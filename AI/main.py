import os
import pathlib
import re

from typing import List
from pathlib import Path

from AI.src.utils import get_last_checkpoint


# def _sorted_checkpoints(
#     output_dir=None, use_mtime=False
# ) -> List[str]:
#     ordering_and_checkpoint_path = []
#
#     glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}**") if os.path.isdir(x)]
#     print(glob_checkpoints)
#
#     for path in glob_checkpoints:
#         if use_mtime:
#             ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
#         else:
#             regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
#             if regex_match is not None and regex_match.groups() is not None:
#                 ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
#
#     checkpoints_sorted = sorted(ordering_and_checkpoint_path)
#     checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
#     print(checkpoints_sorted)
#
#
#     # Make sure we don't delete the best model.
#     if (
#         self.state.best_model_checkpoint is not None
#         and str(Path(self.state.best_model_checkpoint)) in checkpoints_sorted
#     ):
#         best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
#         for i in range(best_model_index, len(checkpoints_sorted) - 2):
#             checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
#     return checkpoints_sorted


def main() -> None:
    # _sorted_checkpoints(
    #     r"D:\Local\Source\Python\semester_9\AIP391\Video_anomaly_detection\AI\results\AIP391\single\train\run1"
    # )
    from AI.src.tools.train import main
    return None


if __name__ == '__main__':
    main()