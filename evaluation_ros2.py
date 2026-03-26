import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def get_latest_run_dir():
    """Automatically finds the newest run folder in the results directory."""
    RESULTS_DIR = '/workspace/results'
    if not os.path.exists(RESULTS_DIR):
        return None
    existing_runs = [d for d in os.listdir(RESULTS_DIR) if d.startswith('run_')]
    if not existing_runs:
        return None
    run_nums = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
    latest_run = max(run_nums)
    return f"/workspace/results/run_{latest_run}"

def run_evaluation(run_dir):
    print(f"\n--- Running Full ROS 2 Evaluation Suite in {run_dir} ---")
    os.makedirs(f'{run_dir}/plots', exist_ok=True)

    # 1. Load Data
    try:
        odom_df = pd.read_csv(f'{run_dir}/odom_data.csv').sort_values('Timestamp')
        ekf_df = pd.read_csv(f'{run_dir}/ekf_data.csv').sort_values('Timestamp')
    except FileNotFoundError:
        print("Error: Could not find CSV files. Make sure the ROS 2 Logger finished successfully.")
        return

    # Merge GT with Estimated State
    merged = pd.merge_asof(ekf_df, odom_df, on='Timestamp', direction='nearest').dropna()

    # Calculate Position Error
    merged['error_x'] = merged['Est_X'] - merged['Loc_X']
    merged['error_y'] = merged['Est_Y'] - merged['Loc_Y']
    merged['pos_error'] = np.sqrt(merged['error_x']**2 + merged['error_y']**2)

    # --- Calculate Jerk ---
    dt = merged['Timestamp'].diff()
    dt = dt.where(dt > 0.01, 0.01) # Force tiny dt values to 0.01 to prevent math explosions
    
    # 1. Smooth the velocity
    smoothed_vel = merged['GT_Velocity'].rolling(window=5, min_periods=1).mean()
    
    # 2. Calculate Accel and Jerk
    merged['Long_Accel'] = smoothed_vel.diff() / dt
    merged['Long_Jerk'] = merged['Long_Accel'].diff() / dt
    
    # Lateral
    yaw_rate_rad = np.radians(merged['Yaw_Degrees'].diff() / dt)
    merged['Lat_Accel'] = smoothed_vel * yaw_rate_rad
    merged['Lat_Jerk'] = merged['Lat_Accel'].diff() / dt

    # 3. Settling Time Fix (Ignore first 2 seconds)
    start_time = merged['Timestamp'].min()
    valid_data = merged[merged['Timestamp'] > (start_time + 2.0)]
    abs_jerk = valid_data['Long_Jerk'].abs().dropna()

    # --- PLOT 1: Trajectory Tracking ---
    margin = 5.0  
    plt.figure(figsize=(10, 8))
    plt.plot(odom_df['Loc_X'], odom_df['Loc_Y'], 'k--', label='Ground Truth', linewidth=2)
    plt.plot(ekf_df['Est_X'], ekf_df['Est_Y'], 'b-', label='ROS 2 EKF Estimate', linewidth=2)
    plt.xlim(min(odom_df['Loc_X']) - margin, max(odom_df['Loc_X']) + margin)
    plt.ylim(min(odom_df['Loc_Y']) - margin, max(odom_df['Loc_Y']) + margin)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Autonomous ROS 2 Trajectory Tracking')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.savefig(f'{run_dir}/plots/1_trajectory_comparison.png')
    plt.close()

    # --- PLOT 2: Error Over Time ---
    plt.figure(figsize=(10, 4))
    plt.plot(merged['Timestamp'], merged['pos_error'], 'r', label='Position Error')
    plt.title('Localization Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{run_dir}/plots/2_error_over_time.png')
    plt.close()

    # --- PLOT 3: Longitudinal Jerk ---
    plt.figure(figsize=(10, 4))
    plt.plot(merged['Timestamp'], merged['Long_Jerk'], color='orange', alpha=0.8)
    plt.axhline(y=3, color='r', linestyle='--', alpha=0.5, label='Comfort Limit (±3)')
    plt.axhline(y=-3, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=10, color='purple', linestyle=':', label='Safety Limit (±10)')
    plt.axhline(y=-10, color='purple', linestyle=':')
    plt.title('Longitudinal Jerk Over Time')
    plt.ylim(-15, 15)
    plt.legend()
    plt.savefig(f'{run_dir}/plots/3_jerk_plot.png')
    plt.close()

    # --- PLOT 4: Jerk Heatmap (Lateral vs Longitudinal) ---
    plt.figure(figsize=(8, 6))
    plt.hist2d(merged['Lat_Jerk'].fillna(0), merged['Long_Jerk'].fillna(0), bins=50, cmap='inferno')
    plt.colorbar(label='Frequency')
    plt.title('2D Jerk Heatmap (ROS 2 Control)')
    plt.xlabel('Lateral Jerk (m/s³)')
    plt.ylabel('Longitudinal Jerk (m/s³)')
    plt.savefig(f'{run_dir}/plots/4_jerk_heatmap.png')
    plt.close()

    # --- PLOT 5: EKF Error Map ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(merged['Est_X'], merged['Est_Y'], c=merged['pos_error'], cmap='inferno', s=20)
    plt.plot(odom_df['Loc_X'], odom_df['Loc_Y'], 'k--', alpha=0.3, label='Ground Truth')
    plt.colorbar(scatter, label='Position Error (m)')
    plt.title('ROS 2 EKF Error Map (Trajectory colored by Error)')
    plt.legend()
    plt.savefig(f'{run_dir}/plots/5_ekf_error_map.png')
    plt.close()

    # --- Metrics Export ---
    duration = merged['Timestamp'].max() - merged['Timestamp'].min()
    
    metrics = {
        "duration_s": float(duration),
        "samples": len(merged),
        "rmse_pos": float(np.sqrt(np.mean(merged['pos_error']**2))),
        "max_error": float(merged['pos_error'].max()),
        "mean_error": float(merged['pos_error'].mean()),
        "avg_speed": float(merged['GT_Velocity'].mean()),
        "max_speed": float(merged['GT_Velocity'].max()),
        "avg_jerk": float(abs_jerk.mean()) if not abs_jerk.empty else 0.0,
        "max_jerk": float(abs_jerk.max()) if not abs_jerk.empty else 0.0,
        "rms_jerk": float(np.sqrt(np.mean(abs_jerk**2))) if not abs_jerk.empty else 0.0,
    }
    
    with open(f'{run_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Generated 5 plots in {run_dir}/plots/")
    print(f"✅ Evaluated {int(metrics['samples'])} samples over {metrics['duration_s']:.1f}s")
    print(f"✅ Final RMSE: {metrics['rmse_pos']:.3f}m | Max Jerk: {metrics['max_jerk']:.3f} m/s³")

if __name__ == '__main__':
    # Auto-find the newest run directory
    latest_run = get_latest_run_dir()
    if latest_run:
        run_evaluation(latest_run)
    else:
        print("No run directories found in results/. Run the ROS 2 simulation first!")