from Delft3D import *
from scipy.optimize import differential_evolution

if __name__ == '__main__':
    config = "config.ini"
    # modify bct and bcc files according to config.ini
    os.system("C:/Users/Carlisle/Anaconda3/python.exe Edit_bcc_bct.py")
    # Generate delft3d runner
    delft3d = Delft3D(config)
    # run algorihtm
    bounds = [(505465, 506557), (2496786, 2497709), (0, 1), (300, 2000)]
    result = differential_evolution(delft3d.solve, config=config, bounds=bounds,
                                    updating='deferred', workers=30, tol=0.0001, disp=True)
    print(result)
