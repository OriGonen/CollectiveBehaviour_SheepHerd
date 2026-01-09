import numpy as np
import matplotlib.pyplot as plt
from utils.utils import load_simulation_results
import sys
import os

def plot_ftm_results(filename):
    results = load_simulation_results(filename)
    
    if 'model' not in results or results['model'] != 'FTM':
        print("Warning: This script is intended for FTM results.")
    
    n_iter = results['params']['n_iter']
    no_shp = results['params']['no_shp']
    
    # Calculate speeds for sheep
    # vel_s shape: (n_runs, n_iter, no_shp, 2)
    spd_s = results['spd_s']
    mean_spd_s = np.mean(spd_s, axis=0) # (n_iter, no_shp)
    
    # Dog speed
    # spd_d shape: (n_runs, n_iter)
    spd_d = results['spd_d']
    mean_spd_d = np.mean(spd_d, axis=0) # (n_iter,)
    
    # Fatigue values sheep
    # M_A_s, M_F_s, M_R_s, TL_s shapes: (n_runs, n_iter, no_shp)
    mean_M_A_s = np.mean(results['M_A_s'], axis=0)
    mean_M_F_s = np.mean(results['M_F_s'], axis=0)
    mean_M_R_s = np.mean(results['M_R_s'], axis=0)
    mean_TL_s = np.mean(results['TL_s'], axis=0)
    
    # Fatigue values dog
    # M_A_d, M_F_d, M_R_d, TL_d shapes: (n_runs, n_iter)
    mean_M_A_d = np.mean(results['M_A_d'], axis=0)
    mean_M_F_d = np.mean(results['M_F_d'], axis=0)
    mean_M_R_d = np.mean(results['M_R_d'], axis=0)
    mean_TL_d = np.mean(results['TL_d'], axis=0)
    
    iterations = np.arange(n_iter)
    
    # 1. Plot sheep speed
    plt.figure(figsize=(10, 6))
    for i in range(no_shp):
        plt.plot(iterations, mean_spd_s[:, i], label=f'Sheep {i+1}')
    plt.xlabel('Iteration (s)')
    plt.ylabel('Sheep Speed')
    plt.title('Mean Sheep Speed over Iterations')
    # plt.legend() # Might be too many sheep
    plt.grid(True)
    plt.show()
    
    # 2. Plot dog speed
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_spd_d, color='black', linewidth=2)
    plt.xlabel('Iteration (s)')
    plt.ylabel('Dog Speed')
    plt.title('Mean Dog Speed over Iterations')
    plt.grid(True)
    plt.show()
    
    # 3. Plot M_F, M_A, M_R of sheep
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    for i in range(no_shp):
        axs[0].plot(iterations, mean_M_A_s[:, i])
        axs[1].plot(iterations, mean_M_F_s[:, i])
        axs[2].plot(iterations, mean_M_R_s[:, i])
    axs[0].set_ylabel('M_A')
    axs[0].set_title('Mean Sheep Fatigue states (M_A, M_F, M_R)')
    axs[1].set_ylabel('M_F')
    axs[2].set_ylabel('M_R')
    axs[2].set_xlabel('Iteration (s)')
    for ax in axs:
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 4. Fatigue values for dog
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_M_A_d, label='M_A')
    plt.plot(iterations, mean_M_F_d, label='M_F')
    plt.plot(iterations, mean_M_R_d, label='M_R')
    plt.xlabel('Iteration (s)')
    plt.ylabel('Fatigue State Value')
    plt.title('Mean Dog Fatigue states')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 5. Plot TL values for sheep
    plt.figure(figsize=(10, 6))
    for i in range(no_shp):
        plt.plot(iterations, mean_TL_s[:, i], label=f'Sheep {i+1}')
    plt.xlabel('Iteration (s)')
    plt.ylabel('TL Value')
    plt.title('Mean Sheep TL Values')
    plt.grid(True)
    plt.show()
    
    # 6. Plot TL values for dog
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_TL_d, color='red', linewidth=2)
    plt.xlabel('Iteration (s)')
    plt.ylabel('TL Value')
    plt.title('Mean Dog TL Values')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try to find a default file if none provided
        data_dir = '../data'
        files = [f for f in os.listdir(data_dir) if f.startswith('ftm_') and f.endswith('.npz')]
        if files:
            filename = os.path.join(data_dir, sorted(files)[-1])
            print(f"No file provided, using latest: {filename}")
            plot_ftm_results(filename)
        else:
            print("Please provide a .npz file path.")
    else:
        plot_ftm_results(sys.argv[1])
