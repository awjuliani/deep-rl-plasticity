from shared.runner import run_experiments
import json
import yaml
import argparse
import datetime

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--out_dir", help="output directory", type=str, default="."
    )
    parser.add_argument("-s", "--seed", help="random seed", type=int, default=0)
    parser.add_argument("-n", "--name", help="experiment name", type=str, default="")
    parser.add_argument("-c", "--condition", help="condition", type=str, default="")
    parser.add_argument(
        "-p",
        "--hyperparams",
        help="hyperparameters",
        type=str,
        default="hyperparams.yaml",
    )
    args = parser.parse_args()
    if args.name == "":
        # set name using date/time
        args.name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # load hyperparameters from yaml file
    with open(args.hyperparams) as f:
        hyperparams = yaml.load(f, Loader=yaml.FullLoader)
    print(json.dumps(hyperparams, indent=4))
    hyperparams["experiment"]["output_dir"] = args.out_dir
    hyperparams["experiment"]["seed"] = args.seed
    hyperparams["experiment"]["name"] = args.name
    if (
        args.condition != ""
        and args.condition in hyperparams["experiment"]["conditions"].keys()
    ):
        hyperparams["experiment"]["conditions"] = {
            args.condition: hyperparams["experiment"]["conditions"][args.condition]
        }
    # run experiments
    run_experiments(hyperparams)
