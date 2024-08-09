import os 
from ts_EPR.utils import make_anim


def main():
    dir_simu = os.path.join('..', 'simu', 'perlin')
    dir_data = os.path.join(dir_simu, 'data')
    dir_simu_output = os.path.join(dir_simu, 'output')
    shifts =  [0.25,0.5,0.75,1.0,1.5,2.0]
    corr_width = 3.5
    set = '35'
    for shift in shifts:
        print('shift: ',shift)
        data_param = "simplex_noise_" + str(corr_width) + '_' + set + "_2_0.5_36_std_1.0_1.5_" + str(shift) + "_0.75_0.5_8.0_drive_10.0_30.0_transfer_50.0_0.25"
        dir_simu_data = os.path.join(dir_data,data_param)
        filename_rate = os.path.join(dir_simu_data, "rate_simplex_noise_baseline_0.bn")
        dir_output_data = os.path.join(dir_simu_output,data_param)
        filename_gif = os.path.join(dir_output_data, 'rate_shift' + str(shift) + '_seed0.gif')

        make_anim(filename_rate, filename_gif, rows = 36, tstart=200, cmap='OrRd')


if __name__ == '__main__':
    main()