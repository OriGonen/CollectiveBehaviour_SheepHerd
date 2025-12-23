from pathlib import Path
from animation import HerdingAnimation
from utils.utils import load_matlab_herding_data, transform_matlab_single_run


if __name__ == "__main__":
    matlab_file = "../data/hm_1_14.mat"

    if not Path(matlab_file).exists():
        print(f"Error: {matlab_file} not found")
        exit(1)

    data = load_matlab_herding_data(matlab_file)

    pos_s, pos_d, vel_s, vel_d, spd_d = transform_matlab_single_run(
        data['pos_s'],
        data['pos_d'],
        data['vel_s'],
        data['vel_d'],
        data['spd_d'],
        run_idx=0
    )

    print(spd_d[0])
    print(data['spd_d'].shape)
    print(f"Loaded Matlab data")

    anim = HerdingAnimation(
        pos_s, pos_d, vel_s, vel_d,
        dog_speeds_log=spd_d,
        show_metrics=True
    )

    anim.run()