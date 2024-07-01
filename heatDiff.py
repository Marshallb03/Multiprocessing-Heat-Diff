import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


def initialize_grid(start_temp, size, init_type='constant'):
    grid = np.zeros((size, size))
    if init_type == 'constant':
        grid[:, 0] = start_temp  # Set leftmost column to starting temperature
    elif init_type == 'random':
        grid = np.random.randint(0, start_temp, size=(size, size))
    elif init_type == 'quarters':
        grid[:int(size/2), :int(size/2)] = start_temp
        grid[int(size/2):, int(size/2):] = start_temp/2
        grid[:int(size/2), int(size/2):] = start_temp/4

    elif init_type == 'walls':
        grid[:, 0] = start_temp
        grid[0, :] = start_temp
        grid[-1, :] = start_temp
        grid[:, -1] = start_temp
    # Add more conditions for different initialization types if needed
    return grid


def update_mini_grid(prev_grid, i, mini_grid_size):
    mini_grid = prev_grid[i*mini_grid_size:(i+1)*mini_grid_size, :]
    new_mini_grid = mini_grid.copy()
    rows, cols = mini_grid.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):

            new_mini_grid[i, j] = 0.25 * (mini_grid[i-1, j] + mini_grid[i+1, j] +
                                          mini_grid[i, j-1] + mini_grid[i, j+1])  #Jacobi method
    return new_mini_grid

def update_grid(prev_grid, current_grid, pool):
    steps = prev_grid.size//cpu_count()
    results = [pool.apply_async(update_mini_grid, args=(prev_grid, i, steps)) for i in range(cpu_count())] # Parallelize
    for i, p in enumerate(results):
        current_grid[i*steps:(i+1)*steps, :] = p.get()

if __name__ == "__main__":
    cmap = str(input("Enter colormap (hot, autumn, etc.): "))
    size = int(input("Enter the size of the grid: "))
    start_temp = int(input("Enter the starting temperature: "))
    
    print("Choose initialization type:")
    print("1. Constant Left Wall")
    print("2. Random")
    print("3. Quarters")
    print("4. Walls")
    init_choice = int(input("Enter your choice (1, 2, 3, 4): "))
    
    if init_choice == 1:
        init_type = 'constant'
    elif init_choice == 2:
        init_type = 'random'
    elif init_choice == 3:
        init_type = 'quarters'
    elif init_choice == 4:
        init_type = 'walls'
    else:
        print("Invalid choice. Using constant initialization by default.")
        init_type = 'constant'
    
    prev_grid = initialize_grid(start_temp, size, init_type)
    current_grid = prev_grid.copy()

    pool = Pool(processes=cpu_count())

    threshold = 0
    max_temp_change = start_temp

    fig, ax = plt.subplots()
    ax.imshow(current_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=start_temp)
    
    plt.title("Heat Distribution Grid")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.waitforbuttonpress()

    # Halt the program when the window gets closed
    def on_close(event):
        print("Closed")
        quit()
    fig.canvas.mpl_connect('close_event', on_close)

    while max_temp_change >= threshold:
        update_grid(prev_grid, current_grid, pool)
        ax.clear()
        ax.imshow(current_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=start_temp)
        plt.pause(0.01)
        max_temp_change = np.max(np.abs(current_grid - prev_grid))
        prev_grid, current_grid = current_grid, prev_grid

    pool.close()
    pool.join()

    print("Final heat distribution grid:")
    np.savetxt('final_grid.txt', current_grid, fmt='%d')

