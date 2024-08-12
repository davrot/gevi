import numpy as np
import os
import json
from jsmin import jsmin  # type:ignore
import argh
from functions.get_trials import get_trials
from functions.get_experiments import get_experiments


def loader(
    filename: str = "config_M_Sert_Cre_49.json", fpath: str = "/data_1/hendrik/gevi"
) -> None:

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

    experiments = get_experiments(raw_data_path).numpy()
    n_exp = experiments.shape[0]

    for i_exp in range(0, n_exp):
        trials = get_trials(raw_data_path, experiments[i_exp]).numpy()
        n_tri = trials.shape[0]

        for i_tri in range(0, n_tri):

            experiment_name: str = (
                f"Exp{experiments[i_exp]:03d}_Trial{trials[i_tri]:03d}"
            )
            tmp_fname = os.path.join(
                fpath,
                "output_" + config["mouse_identifier"],
                experiment_name + "_acceptor_donor.npz",
            )
            print(f'Processing file "{tmp_fname}"...')
            tmp = np.load(tmp_fname)

            tmp_data_sequence = tmp["data_donor"]
            tmp_light_signal = tmp["data_acceptor"]

            if (i_exp == 0) and (i_tri == 0):
                mask = tmp["mask"]
                new_shape = [n_exp, *tmp_data_sequence.shape]
                data_sequence = np.zeros(new_shape)
                light_signal = np.zeros(new_shape)

            # Here you might want to use the exp fit and removal...
            data_sequence[i_exp] += tmp_data_sequence / n_tri
            light_signal[i_exp] += tmp_light_signal / n_tri

    np.save("dsq_" + config["mouse_identifier"], data_sequence)
    np.save("lsq_" + config["mouse_identifier"], light_signal)
    np.save("msq_" + config["mouse_identifier"], mask)


if __name__ == "__main__":
    argh.dispatch_command(loader)
