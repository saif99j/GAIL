import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Circle

def plot_reward(env_name, algo_name, window_len_smooth=50, min_window_len_smooth=1, 
                window_len_var=5, min_window_len_var=1, linewidth_smooth=1.5, 
                linewidth_var=2, alpha_smooth=1, alpha_var=0.1, colors=None):
    """
    Plot the reward curve from the training log
    
    Args:
        env_name: environment name
        algo_name: algorithm name
        window_len_smooth: window length for smoothing
        min_window_len_smooth: minimum window length for smoothing
        window_len_var: window length for variance
        min_window_len_var: minimum window length for variance
        linewidth_smooth: linewidth for smooth curve
        linewidth_var: linewidth for variance curve
        alpha_smooth: alpha for smooth curve
        alpha_var: alpha for variance curve
        colors: list of colors for plotting
    """
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson', 'gray', 'black']
        
    # read data
    log_f = algo_name + '_' + env_name + "_log.csv"
    data = pd.read_csv(log_f)

    ax = plt.gca()

    # smooth data
    data['reward_smooth'] = data['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
    data['reward_var'] = data['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

    # plot the lines
    data.plot(kind='line', x='episode', y='reward_smooth', ax=ax, color=colors[0], linewidth=linewidth_smooth, alpha=alpha_smooth)
    data.plot(kind='line', x='episode', y='reward_var', ax=ax, color=colors[0], linewidth=linewidth_var, alpha=alpha_var)

    # grid of figure
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

    # x and y axis setting
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Rewards", fontsize=12)

    # figure title setting
    plt.title(env_name + "_" + algo_name + "_reward", fontsize=14)
    
    return ax

def plot_timestep(env_name, algo_name, window_len_smooth=50, min_window_len_smooth=1, 
                 window_len_var=5, min_window_len_var=1, linewidth_smooth=1.5, 
                 linewidth_var=2, alpha_smooth=1, alpha_var=0.1, colors=None):
    """
    Plot the timestep curve from the training log
    
    Args:
        env_name: environment name
        algo_name: algorithm name
        window_len_smooth: window length for smoothing
        min_window_len_smooth: minimum window length for smoothing
        window_len_var: window length for variance
        min_window_len_var: minimum window length for variance
        linewidth_smooth: linewidth for smooth curve
        linewidth_var: linewidth for variance curve
        alpha_smooth: alpha for smooth curve
        alpha_var: alpha for variance curve
        colors: list of colors for plotting
    """
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson', 'gray', 'black']
    
    # read data
    log_f = algo_name + '_' + env_name + "_log.csv"
    data = pd.read_csv(log_f)
    
    ax = plt.gca()

    # smooth data
    data['timestep_smooth'] = data['timestep'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
    data['timestep_var'] = data['timestep'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

    # plot the lines
    data.plot(kind='line', x='episode', y='timestep_smooth', ax=ax, color=colors[1], linewidth=linewidth_smooth, alpha=alpha_smooth)
    data.plot(kind='line', x='episode', y='timestep_var', ax=ax, color=colors[1], linewidth=linewidth_var, alpha=alpha_var)

    # grid of figure
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

    # x and y axis setting
    ax.set_xlabel("Episodes", fontsize=12)
    ax.set_ylabel("Timesteps", fontsize=12)

    # figure title setting
    plt.title(env_name + "_" + algo_name + "_timestep", fontsize=14)
    
    return ax

def plot_traj(env_name, algo_name, episode=1450, interval=80, radius=0.15, colors=None):
    """
    Plot the trajectory of the agent
    
    Args:
        env_name: environment name
        algo_name: algorithm name
        episode: which episode to plot
        interval: how much circle to draw
        radius: radius of circle
        colors: list of colors for plotting
    """
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson', 'gray', 'black']

    # read data
    log_f = env_name + "_" + algo_name + "_rollout.csv"
    data = pd.read_csv(log_f, sep='\t')
    data = data[data.episode==episode]
    
    # num is the number of ships
    if not data.empty:
        ls = eval(data.iloc[0]['position'])
    else:
        print("DataFrame is empty!")
        return None

    num = len(ls)
    # x is a list of list, every list inside is x coordinate of ship 
    x = [[] for i in range(num)]
    # y is a list of list, every list inside is y coordinate of ship 
    y = [[] for i in range(num)]
    # get x and y of every ship
    for idx, row in data.iterrows():
        position = row['position']
        position = eval(position)
        for i in range(num):
            x[i].append(position[i][0])
            y[i].append(position[i][1])

    ax = plt.gca()

    # set alpha
    step = (0.5 - 0.1) / interval
    alphas = np.arange(0.1, 0.5, step)
    alphas = np.append(alphas, [0.8] * len(x[0]))
    step = int(len(x[0]) / interval)

    # plot
    for i in range(num):
        idx = 0
        for j in range(0, len(x[i]), step):
            if i == 0: color = 'blue'
            else: color = 'black'
            circle = Circle(xy=(x[i][j], y[i][j]), color=color, alpha=alphas[idx], radius=radius, linewidth=0)
            ax.add_patch(circle)
            idx += 1

    # x and y label
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.axis("equal")

    # title of figure
    plt.title(env_name + "_" + algo_name + "_trajectory", fontsize=14)
    
    return ax

def plot_cpa(env_name, algo_name, episode=1450, colors=None):
    """
    Plot the Closest Point of Approach (CPA) data
    
    Args:
        env_name: environment name
        algo_name: algorithm name
        episode: which episode to plot
        colors: list of colors for plotting
    """
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson', 'gray', 'black']
    
    # read data
    log_f = env_name + "_" + algo_name + "_rollout.csv"
    data = pd.read_csv(log_f, sep='\t')
    data = data[data.episode==episode]
    
    # num is the number of ship
    if not data.empty:
        ls = eval(data.iloc[0]['position'])
    else:
        print("DataFrame is empty!")
        return None
        
    num = len(ls)
    # list of list, every list inside is the information of a ship
    dcpas = [[] for i in range(num)]
    tcpas = [[] for i in range(num)]
    drs = [[] for i in range(num)]
    vrs = [[] for i in range(num)]
    phios = [[] for i in range(num)]
    phits = [[] for i in range(num)]
    phirs = [[] for i in range(num)]
    # actions of the agent ship
    acts = []
    # get information and calculation
    for idx, row in data.iterrows():
        position = row['position']
        position = eval(position)
        orientation = row['orientation']
        orientation = eval(orientation)
        speed = row['speed']
        speed = eval(speed)
        for i in range(1, num):
            dx = position[i][0] - position[0][0]
            dy = position[i][1] - position[0][1]
            phio = -orientation[0]+math.pi/2
            phit = -orientation[i]+math.pi/2
            vo = speed[0]
            vt = speed[i]
            vr = vo * math.sqrt(1 + (vt/vo)**2 - 2*(vt/vo)*math.cos(phio-phit))
            phir = math.acos((vo-vt*math.cos(phio-phit)) / vr)
            dr = math.sqrt(dx**2 + dy**2)
            ar = math.asin(dx/dr)
            dcpa = dr * np.sin(phir-ar-math.pi)
            tcpa = dr * np.cos(phir-ar-math.pi) / vr
            dcpas[i].append(dcpa)
            tcpas[i].append(tcpa)
            drs[i].append(dr)
            vrs[i].append(vr)
            phios[i].append(phio)
            phits[i].append(phit)
            phirs[i].append(phir)
        act = row['action']
        act = eval(act)[0] - 1
        act *= 10
        acts.append(act)

    # figure 1: DCPA
    plt.figure(figsize=(10,6))
    plt.axhline(0.0, linestyle='--', label='min_dcpa', color='black')
    for i in range(1, num):
        length = len(dcpas[i])
        plt.plot(np.arange(0, length, 1), dcpas[i], label='OS-TS{}'.format(i), color=colors[i-1])
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("DCPA", fontsize=12)
    plt.legend()
    plt.title(env_name + "_" + algo_name + "_DCPA", fontsize=14)
    
    # figure 2: Distance
    plt.figure(figsize=(10,6))
    plt.axhline(0.6, linestyle='--', label='min_distance = 0.6', color='black')
    for i in range(1, num):
        length = len(drs[i])
        plt.plot(np.arange(0, length, 1), drs[i], label='OS-TS{}'.format(i), color=colors[i-1])
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.legend()
    plt.title(env_name + "_" + algo_name + "_Distance", fontsize=14)
    
    # figure 3: Relative velocity
    plt.figure(figsize=(10,6))
    for i in range(1, num):
        length = len(vrs[i])
        plt.plot(np.arange(0, length, 1), vrs[i], label='OS-TS{}'.format(i), color=colors[i-1])
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Vr", fontsize=12)
    plt.legend()
    plt.title(env_name + "_" + algo_name + "_Relative_Velocity", fontsize=14)
    
    # figure 4: Phi angles
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(0, length, 1), phios[1], label="OS", color=colors[0])
    for i in range(1, num):
        length = len(phirs[i])
        plt.plot(np.arange(0, length, 1), phits[i], label="TS{}".format(i), color=colors[i])
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Phi", fontsize=12)
    plt.legend()
    plt.title(env_name + "_" + algo_name + "_Phi", fontsize=14)

    # figure 5: TCPA
    plt.figure(figsize=(10,6))
    plt.axhline(0, linestyle='--', label='min_tcpa', color='black')
    for i in range(1, num):
        length = len(dcpas[i])
        plt.plot(np.arange(0, length, 1), tcpas[i], label='OS-TS{}'.format(i), color=colors[i-1])
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("TCPA", fontsize=12)
    plt.legend()
    plt.title(env_name + "_" + algo_name + "_TCPA", fontsize=14)
    
    return plt.gcf()

def main():
    # Default parameters
    fig_width = 10
    fig_height = 6

    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    window_len_smooth = 50
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson', 'gray', 'black']

    # Set up the algorithm and environment to plot
    algo_name = 'ppo'  # or 'gail'
    env_name = 'env4'

    # Create the reward plot
    plt.figure(figsize=(fig_width, fig_height))
    plot_reward(env_name, algo_name, window_len_smooth, min_window_len_smooth, 
                window_len_var, min_window_len_var, linewidth_smooth, 
                linewidth_var, alpha_smooth, alpha_var, colors)
    plt.savefig(f"{env_name}_{algo_name}_reward.png")

    # Create the timestep plot
    plt.figure(figsize=(fig_width, fig_height))
    plot_timestep(env_name, algo_name, window_len_smooth, min_window_len_smooth, 
                 window_len_var, min_window_len_var, linewidth_smooth, 
                 linewidth_var, alpha_smooth, alpha_var, colors)
    plt.savefig(f"{env_name}_{algo_name}_timestep.png")

    # Create the trajectory plot
    plt.figure(figsize=(fig_width, fig_height))
    plot_traj(env_name, algo_name, episode=1450, interval=80, radius=0.15, colors=colors)
    plt.savefig(f"{env_name}_{algo_name}_trajectory.png")

    # Create the CPA plots
    plot_cpa(env_name, algo_name, episode=1450, colors=colors)
    plt.savefig(f"{env_name}_{algo_name}_cpa.png")

    plt.show()

if __name__ == "__main__":
    main()