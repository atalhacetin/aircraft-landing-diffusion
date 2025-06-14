import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import os # For creating directories and path manipulation

def plot_saved_solutions_with_animation(filepath="landing_mpc_dataset.npz"):
    """
    Loads solutions from a .npz file, animates each trajectory, and saves the animations.

    Args:
        filepath (str): The path to the .npz file containing the saved data.
    """
    try:
        data = np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please ensure the dataset is generated.")
        return

    # Extract data
    X_list = data['X']
    U_list = data['U']
    centers_list = data['centers'] # These are the base_centers for each scenario
    radii_list = data['radii']     # These are the base_radii for each scenario
    car_velocities = data['car_velocities']

    print(f"Loaded {len(X_list)} successful trajectories from {filepath}")

    # Define common plot parameters
    # These should match your OCP setup (Tf=1.0, N=10)
    dt = 1.0 / 10 

    # Create directory for animations if it doesn't exist
    animations_dir = "animations"
    os.makedirs(animations_dir, exist_ok=True)
    print(f"Saving animations to: {os.path.abspath(animations_dir)}")

    # Loop through each successful scenario to create and save an animation
    for i, (X_log, U_log, centers_base, radii_base, car_vel) in enumerate(zip(X_list, U_list, centers_list, radii_list, car_velocities)):
        print(f"\n--- Animating Scenario {i+1} (Car Vel: {car_vel} m/s) ---")

        # num_frames will be sim_steps_per_run + 1 (states logged)
        num_frames = X_log.shape[0] 

        # --- Figure setup for animation ---
        fig_anim = plt.figure(figsize=(10, 8))
        ax_anim  = fig_anim.add_subplot(111, projection='3d')
        ax_anim.set_xlabel('x [m]')
        ax_anim.set_ylabel('y [m]')
        ax_anim.set_zlabel('h [m]')
        ax_anim.set_title(f'MPC Trajectory (Scenario {i+1}, Car Vel: {car_vel} m/s)')

        # Unpack the radii once per scenario (since all obstacles share the same global_radii)
        rx, ry, rz = radii_base 

        # --- Dynamic axis limits calculation ---
        # Include initial and final obstacle positions in the range test
        # The obstacles shift by car_vel * t_final_simulation (last frame time)
        final_shift_x = car_vel * (num_frames - 1) * dt

        # Collect all x, y, z coordinates for auto-scaling
        all_x_coords = np.hstack([X_log[:, 0], centers_base[:, 0] + final_shift_x, centers_base[:, 0]])
        all_y_coords = np.hstack([X_log[:, 1], centers_base[:, 1]])
        all_z_coords = np.hstack([X_log[:, 2], centers_base[:, 2] + rz, centers_base[:, 2] - rz])

        # Calculate max range and mid points for equal aspect ratio
        max_range = np.array([all_x_coords.ptp(), all_y_coords.ptp(), all_z_coords.ptp()]).max() / 2
        mid_x     = (all_x_coords.min() + all_x_coords.max()) * 0.5
        mid_y     = (all_y_coords.min() + all_y_coords.max()) * 0.5
        mid_z     = (all_z_coords.min() + all_z_coords.max()) * 0.5

        # Apply limits
        ax_anim.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_anim.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_anim.set_zlim(mid_z - max_range, mid_z + max_range)
        ax_anim.grid(True)

        # --- Obstacle geometry setup for animation ---
        # Create a mesh for a single ellipsoid, which will be scaled and shifted
        u_sphere  = np.linspace(0, 2*np.pi, 24)
        v_sphere  = np.linspace(0, np.pi, 12)
        uu, vv = np.meshgrid(u_sphere, v_sphere)

        surf_handles = [] # List to hold surface plot objects for each obstacle
        for cx_base, cy_base, cz_base in centers_base:
            # Initial position of obstacle (t=0, shift=0)
            xs = rx*np.cos(uu)*np.sin(vv) + cx_base
            ys = ry*np.sin(uu)*np.sin(vv) + cy_base
            zs = rz*np.cos(vv)            + cz_base
            h = ax_anim.plot_surface(xs, ys, zs, color='crimson',
                                    alpha=0.25, linewidth=0)
            surf_handles.append(h)

        # Trajectory line and current position point
        traj_line, = ax_anim.plot([], [], [], lw=2, color='royalblue', label='Aircraft Trajectory')
        point,     = ax_anim.plot([], [], [], 'o', color='navy', markersize=6, label='Aircraft Position')
        
        ax_anim.legend() # Display legend for trajectory elements

        # --- Animation callbacks ---
        def init():
            # Reset aircraft trajectory and point
            traj_line.set_data([], [])
            traj_line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            
            # Reset obstacle positions to their initial (t=0) state for init frame
            # This is crucial for the animation to start correctly
            for idx, (cx_base, cy_base, cz_base) in enumerate(centers_base):
                # Remove the old surface plot before creating a new one
                surf_handles[idx].remove() 
                
                # Create the initial (t=0) surface
                xs = rx*np.cos(uu)*np.sin(vv) + cx_base
                ys = ry*np.sin(uu)*np.sin(vv) + cy_base
                zs = rz*np.cos(vv)            + cz_base
                surf_handles[idx] = ax_anim.plot_surface(
                    xs, ys, zs, color='crimson', alpha=0.25, linewidth=0)
            
            return [traj_line, point, *surf_handles]

        def animate(k):
            # Current time in the simulation
            t = k * dt
            # Calculate the total shift for obstacles at the current time
            shift = car_vel * t

            # Update aircraft trajectory and current position
            traj_line.set_data(X_log[:k+1,0], X_log[:k+1,1])
            traj_line.set_3d_properties(X_log[:k+1,2])
            point.set_data([X_log[k,0]], [X_log[k,1]])
            point.set_3d_properties([X_log[k,2]])

            # Update obstacle positions
            for idx, (cx_base, cy_base, cz_base) in enumerate(centers_base):
                # Remove the previous surface plot
                surf_handles[idx].remove() 
                
                # Create a new surface plot with the current shift applied
                xs = rx*np.cos(uu)*np.sin(vv) + (cx_base + shift)
                ys = ry*np.sin(uu)*np.sin(vv) + cy_base
                zs = rz*np.cos(vv)            + cz_base
                surf_handles[idx] = ax_anim.plot_surface(
                    xs, ys, zs, color='crimson', alpha=0.25, linewidth=0)

            return [traj_line, point, *surf_handles]

        # Set frames per second (fps) for the animation
        fps = int(1/dt) # If dt is 0.1, fps will be 10

        # Create the animation object
        ani = animation.FuncAnimation(fig_anim, animate,
                                    frames=num_frames, # Animate for all recorded state points
                                    init_func=init, blit=False, # blit=False is often needed for 3D plots
                                    interval=1000/fps, repeat=False) # interval in ms, repeat=False for single playback

        # Save the animation to an MP4 file
        animation_filename = os.path.join(animations_dir, f"mpc_landing_scenario_{i+1}_carvel_{car_vel}.mp4")
        print(f"Saving animation to: {animation_filename}")
        ani.save(animation_filename,
                 writer="ffmpeg",        # Requires FFmpeg installed and in your system's PATH
                 fps=fps, dpi=200,       # Set resolution and fps
                 bitrate=-1)            # -1 lets ffmpeg choose a sane bitrate

        # Close the figure to free up memory, especially when generating many animations
        plt.close(fig_anim) 

    print("\nAll animations saved.")


if __name__ == "__main__":
    # Call the new function to generate and save animations
    plot_saved_solutions_with_animation()