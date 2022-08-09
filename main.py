import argparse
from os import listdir
from os.path import join
from typing import Any, Optional, Dict
from metrics_single_band import Evaluation
from clip import Clipper


class ArgumentParser:
    def __init__(self):
        self._args: Optional[Dict[str, Any]] = None
        self._parser = argparse.ArgumentParser(
            prog='WV-S2 benchmark',
            description='WV-S2 dataset build and evaluation scripts',
        )
        self._subparsers = self._parser.add_subparsers(dest='command', required=True)
        self._build_dataset_clipping_command()
        self._build_evaluate()

    @property
    def args(self) -> Dict[str, Any]:
        if self._args is None:
            args = vars(self._parser.parse_args())
            self._args = args
        return self._args

    def _build_dataset_clipping_command(self):
        parser_dataset_clipping = self._subparsers.add_parser('build_dataset')
        parser_dataset_clipping.add_argument('--raw_data_path', type=str, required=True,
                                             help='Path to the raw dataset.')
        parser_dataset_clipping.add_argument('--out_data_path', type=str, required=True,
                                             help='Output dataset path.')
        parser_dataset_clipping.add_argument('--make_rgb_pairs', action='store_false', default=False,
                                             help='If True, clips only RGB channels of WorldView and Sentinel '
                                                  'images and generates pairs, blended and tiled images.')

    def _build_evaluate(self):
        parser_evaluation = self._subparsers.add_parser('evaluate')
        parser_evaluation.add_argument('--config_path', type=str, default="config.yaml",
                                       help='Path to the config file.')
        parser_evaluation.add_argument('--dataset_path', type=str, required=True,
                                       help='Path to the benchmark dataset.')


def main():
    parser = ArgumentParser()
    args = parser.args
    if args["command"] == "build_dataset":
        sentinel_scenes = [join(args["raw_data_path"], scene) for scene in listdir(args["raw_data_path"])
                           if scene[0].isdigit()]
        for scene in sentinel_scenes:
            clipper = Clipper(scene)
            if args["make_rgb_pairs"]:
                clipper.make_rgb_pairs(args["out_data_path"])
            else:
                clipper.clip_full_scene(args["out_data_path"])
    elif args["command"] == "evaluate":
        evaluate = Evaluation(args["config_path"])
        evaluate.calculate_metrics(args["dataset_path"])
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
