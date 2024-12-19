import json
import os
import argh
from jsmin import jsmin  # type:ignore
from functions.get_trials import get_trials
from functions.get_experiments import get_experiments


def converter(config_filename: str = "config.json") -> None:

    filename: str = config_filename

    if os.path.isfile(filename) is False:
        print(f"{filename} is missing")
        exit()

    with open(filename, "r") as file:
        config = json.loads(jsmin(file.read()))

    raw_data_path: str = os.path.join(
        config["basic_path"],
        config["recoding_data"],
        config["mouse_identifier"],
        config["raw_path"],
    )

    if os.path.isdir(raw_data_path) is False:
        print(f"ERROR: could not find raw directory {raw_data_path}!!!!")
        exit()

    experiments = get_experiments(raw_data_path).numpy()

    for experiment in experiments:

        trials = get_trials(raw_data_path, experiment).numpy()
        assert trials.shape[0] > 0

        with open(
            os.path.join(
                raw_data_path,
                f"Exp{experiment:03d}_Trial{trials[0]:03d}_Part001_meta.txt",
            ),
            "r",
        ) as file:
            metadata = json.loads(jsmin(file.read()))

        filename_out: str = f"meta_{config["mouse_identifier"]}_exp{experiment:03d}.json"

        with open(filename_out, 'w') as file:
            json.dump(metadata, file)


if __name__ == "__main__":
    argh.dispatch_command(converter)
