import numpy as np
import os
import argh


# mouse:int = 0, 1, 2, 3, 4
def loader(mouse: int = 0, fpath: str = "/data_1/hendrik/gevi") -> None:

    mouse_name = [
        "M_Sert_Cre_41",
        "M_Sert_Cre_42",
        "M_Sert_Cre_45",
        "M_Sert_Cre_46",
        "M_Sert_Cre_49",
    ]

    n_tris = [
        [
            15,
            15,
            30,
            30,
            30,
            30,
        ],  # 0 in cond 7
        [
            15,
            15,
            30,
            30,
            30,
            30,
        ],  # 0 in cond 7
        [
            15,
            15,
            30,
            30,
            30,
            30,
        ],  # 0 in cond 7
        [
            20,
            40,
            20,
            20,
        ],  # 0, 0, 0 in cond 5-7
        [
            20,
            40,
            20,
            20,
        ],  # 0, 0, 0 in cond 5-7
    ]

    # 41, 42, 45, 46, 49:
    #               "1": "control",
    #               "2": "visual control grating 100 1s",
    #               "3": "optical Stimulation 20Hz 50% 5 Intervals",
    #               "4": "optical Stimulation 20Hz 100% 5 Intervals",
    #               "5": "optical Stimulation 20Hz 50% and grating 100",
    #               "6": "optical Stimulation 20Hz 100% and grating 100",
    #               "7": "grating 3s"

    lbs = [
        [
            "control",
            "visual control",
            "op20 50 5",
            "op20 100 5",
            "op20 50 grat",
            "op20 100 grat",
        ],
        [
            "control",
            "visual control",
            "op20 50 5",
            "op20 100 5",
            "op20 50 grat",
            "op20 100 grat",
        ],
        [
            "control",
            "visual control",
            "op20 50 5",
            "op20 100 5",
            "op20 50 grat",
            "op20 100 grat",
        ],
        ["control", "visual control", "op20 50 5", "op20 100 5"],
        ["control", "visual control", "op20 50 5", "op20 100 5"],
    ]

    n_exp = len(n_tris[mouse])

    for i_exp in range(n_exp):
        n_tri = n_tris[mouse][i_exp]
        for i_tri in range(n_tri):

            experiment_name: str = f"Exp{i_exp + 1:03d}_Trial{i_tri + 1:03d}"
            tmp_fname = os.path.join(
                fpath,
                "output_" + mouse_name[mouse],
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
            data_sequence[i_exp] += tmp_data_sequence / n_tri
            light_signal[i_exp] += tmp_light_signal / n_tri

    np.save("dsq_" + mouse_name[mouse], data_sequence)
    np.save("lsq_" + mouse_name[mouse], light_signal)
    np.save("msq_" + mouse_name[mouse], mask)


if __name__ == "__main__":
    argh.dispatch_command(loader)
