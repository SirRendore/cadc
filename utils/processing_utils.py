import sys

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

# sys.path.append("/home/zhaohaoa/cadc/cadc_devkit") 
CUR_DIR = Path(__file__).parent
sys.path.append(str(CUR_DIR.parent / "cadc_devkit"))


def load_data(dir_path):
    '''
    Each file in dir_path contains one line per timestamp. Load as an np array of arrays.

    '''
    dir_path = Path(dir_path)
    data = np.array([np.loadtxt(f) for f in sorted(dir_path.glob("*.txt"))])
    return data
    

def load_timestamps(file_path):
    '''
    Load timestamps from a file and convert them to seconds relative to the first timestamp. Returns a numpy array of floats and first timestamp.
    '''
    timestamps = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        start_time_str = lines[0].strip()
        start_time = np.datetime64(start_time_str)
        for line in lines:
            time_str = line.strip()
            time = np.datetime64(time_str)
            delta = (time - start_time) / np.timedelta64(1, 's')
            timestamps.append(float(delta))
    return np.array(timestamps), start_time

def resample_hz(poses, times, target_hz):
    '''
    Resample poses to a target frequency using linear interpolation for translation
    and spherical linear interpolation (SLERP) for rotation.
    Args:
        poses: List or array of (4,4) pose matrices.
        times: List or array of timestamps corresponding to each pose.
        target_hz: Desired frequency in Hz for resampling.
    Returns:
        poses_new: Resampled poses at target frequency.
        times_new: Corresponding timestamps for resampled poses.
    '''

    poses = np.asarray(poses, dtype=float)
    times = np.asarray(times, dtype=float)

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (N,4,4), got {poses.shape}")

    if times.ndim != 1 or times.shape[0] != poses.shape[0]:
        raise ValueError("times must be shape (N,) and match poses")

    # Target time grid
    dt = 1.0 / target_hz
    times_new = np.arange(times[0], times[-1], dt)

    # Translation: linear interpolation
    t = poses[:, :3, 3]
    x = np.interp(times_new, times, t[:, 0])
    y = np.interp(times_new, times, t[:, 1])
    z = np.interp(times_new, times, t[:, 2])

    # Rotation: SciPy SLERP
    R_in = R.from_matrix(poses[:, :3, :3])
    slerp = Slerp(times, R_in)
    R_new = slerp(times_new)

    # --- Assemble output poses ---
    poses_new = np.zeros((len(times_new), 4, 4), dtype=float)
    poses_new[:, 3, 3] = 1.0
    poses_new[:, :3, :3] = R_new.as_matrix()
    poses_new[:, :3, 3] = np.column_stack((x, y, z))

    return poses_new, times_new


def plot_poses_3d(poses, arrow_length=1, step=20, ax=None):
    '''
    Plots 3D vehicle trajectory with orientation arrows.
    Args:
        poses: List or array of (4,4) pose matrices.
        arrow_length: Length of the orientation arrows.
        step: Step size for downsampling arrows to reduce clutter.
    
    poses are in the format:
        [ R | t ]
        [ 0 | 1 ]
    '''

    if ax is None:
        fig = ax.get_figure()
        ax = fig.add_subplot(111, projection='3d')
    

    P = np.asarray(poses)
    if P.ndim != 3 or P.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses shaped (N,4,4), got {P.shape}")
    
    t = P[:, 0:3, 3]      
    fwd = P[:, 0:3, 0]    # forward directions (R[:,0])

    # Trajectory 
    ax.plot(t[:, 0], t[:, 1], t[:, 2], ".", label="Vehicle Trajectory")

    # Downsample arrows to avoid clutter
    idx = np.arange(0, t.shape[0], step)
    ts = t[idx]                 # (M,3)
    fs = fwd[idx]               # (M,3)

    # normalize so all arrows same length
    norms = np.linalg.norm(fs, axis=1, keepdims=True)
    fs_unit = fs / np.clip(norms, 1e-12, None)

    # draw all arrows
    ax.quiver(
        ts[:, 0], ts[:, 1], ts[:, 2],          # starts
        fs_unit[:, 0], fs_unit[:, 1], fs_unit[:, 2],  # directions
        length=arrow_length,
        arrow_length_ratio=0.2,
        # normalize=True,
        color='r',
    )

    ax.set_zlim3d(-5,5)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    # ax.legend()
    
    return ax

def convert_to_2d(poses):
    '''
    Conver to [x, y, yaw] 2D representation from 4x4 poses.
    Args:
        poses: List or array of (4,4) pose matrices.
    Returns:
        poses_2d: Array of (N, 3) with [x, y, yaw] for each pose.
    ''' 
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (N,4,4), got {poses.shape}")
    N = poses.shape[0]
    poses_2d = np.zeros((N, 3), dtype=float)  # [x, y, yaw]
    for i in range(N):
        t = poses[i, :3, 3]
        R_mat = poses[i, :3, :3]
        yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])  # yaw from rotation matrix
        poses_2d[i] = [t[0], t[1], yaw]
    return poses_2d


def infer_direction_3d(
    poses,
    *,
    yaw_deg=12.0,
    min_step=1e-3,                 # ignore tiny steps
):
    
    '''
    Given array of poses, infer direction based on start and end heading.
    Return as 1-hot 4D vector (left, forward, right, unknown)
    Args:
        poses: List or array of (4,4) pose matrices.
        yaw_deg: Minimum yaw angle in degrees to consider a turn.
        min_step: Minimum step distance to consider for forward motion.
    '''
    # output order: [left, forward, right, unknown]
    LEFT, FWD, RIGHT, UNK = 0, 1, 2, 3

    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < 2:
        return np.array([0, 0, 0, 1], dtype=np.float32)

    R = poses[:, :3, :3]
    t = poses[:, :3, 3]

    # ---- 1) TURN: accumulate signed yaw from relative rotations (z-up) ----
    total_yaw = 0.0
    for i in range(poses.shape[0] - 1):
        Rrel = R[i].T @ R[i + 1]  # local->world => relative in local at i
        # yaw about +Z (right-hand rule): + => left (CCW), - => right (CW)
        total_yaw += np.arctan2(Rrel[1, 0], Rrel[0, 0])

    total_yaw_deg = np.degrees(total_yaw)
    if total_yaw_deg > yaw_deg:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    if total_yaw_deg < -yaw_deg:
        return np.array([0, 0, 1, 0], dtype=np.float32)

    # ---- 2) STRAIGHT: vote using forward motion in local frame ----
    fwd_sum = 0.0
    used = 0.0
    for i in range(poses.shape[0] - 1):
        dp_world = t[i + 1] - t[i]
        step = float(np.linalg.norm(dp_world))
        if step < min_step:
            continue
        dp_local = R[i].T @ dp_world
        # forward is x
        if dp_local[0] > 0:
            fwd_sum += dp_local[0]
            used += 1.0

    if used > 0 and fwd_sum / used > min_step:
        return np.array([0, 1, 0, 0], dtype=np.float32)

    return np.array([0, 0, 0, 1], dtype=np.float32)


def infer_direction_2d(
    poses,
    *,
    yaw_deg=12.0,
    min_step=1e-3, # ignore tiny steps
):
    '''
    Given array of 2D poses [x, y, yaw], infer direction based on start and end heading.
    Return as 1-hot 4D vector (left, forward, right, unknown)
    Args:
        poses: List or array of 2D poses [x, y, yaw].
        yaw_deg: Minimum yaw angle in degrees to consider a turn.
        min_step: Minimum step distance to consider for forward motion.
    '''
    # output order: [left, forward, right, unknown]
    LEFT, FWD, RIGHT, UNK = 0, 1, 2, 3

    poses = np.asarray(poses)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] < 2:
        return np.array([0, 0, 0, 1], dtype=np.float32)

    t = poses[:, :2]
    yaw = poses[:, 2]

    # ---- 1) TURN: accumulate signed yaw differences ----
    total_yaw = 0.0
    for i in range(poses.shape[0] - 1):
        dyaw = yaw[i + 1] - yaw[i]
        # normalize to [-pi, pi]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        total_yaw += dyaw

    total_yaw_deg = np.degrees(total_yaw)
    if total_yaw_deg > yaw_deg:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    if total_yaw_deg < -yaw_deg:
        return np.array([0, 0, 1, 0], dtype=np.float32)

    # ---- 2) STRAIGHT: vote w forward motion in local frame ----
    fwd_sum = 0.0
    used = 0.0
    for i in range(poses.shape[0] - 1):
        dp_world = t[i + 1] - t[i]
        step = float(np.linalg.norm(dp_world))
        if step < min_step:
            continue
        # local frame forward is along heading
        heading = np.array([np.cos(yaw[i]), np.sin(yaw[i])])
        dp_local_fwd = np.dot(dp_world, heading)
        if dp_local_fwd > 0:
            fwd_sum += dp_local_fwd
            used += 1.0
    if used > 0 and fwd_sum / used > min_step:
        return np.array([0, 1, 0, 0], dtype=np.float32)
    return np.array([0, 0, 0, 1], dtype=np.float32)

def plot_poses_2d(poses, step=20, ax=None, **kwargs):
    '''
    Plot 2D vehicle trajectory with orientation arrows.
    Args:
        poses: List or array of 2D poses [x, y, yaw].
        step: Step size for downsampling arrows to reduce clutter.
    
        ax: matplotlib axis to plot on.
    '''
    P = np.asarray(poses)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected poses shaped (N,3), got {P.shape}")
    t = P[:, 0:2]            # positions (N,2)
    yaw = P[:, 2]            # headings (N,)
    fwd = np.column_stack((np.cos(yaw), np.sin(yaw)))  # forward directions (N,2)
    

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # Trajectory
    ax.plot(t[:, 0], t[:, 1], ".", **kwargs)
    
    # Arrows
    if step > 0:
        idx = np.arange(0, t.shape[0], step)
        ts = t[idx]                 # (M,3)
        fs = fwd[idx]               # (M,3)
        # draw all arrows
        ax.quiver(
            ts[:, 0], ts[:, 1],          # starts
            fs[:, 0], fs[:, 1],  # directions
            color='r',
        )
    
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.axis('equal')
    return ax


def find_closest_timestamp_inds(target_times, timestamps):
    '''
    Find indices of the closest timestamps in 'timestamps' for each time in 'target_times'.
    Args:
        target_times (np.ndarray): Array of target timestamps to find closest matches for.
        timestamps (np.ndarray): Array of timestamps to search within.
    Returns:
        np.ndarray: Array of indices of the closest timestamps in 'timestamps' for each target time.
    '''
    idx = np.searchsorted(timestamps, target_times)
    idx = np.clip(idx, 1, len(timestamps)-1)
    left = idx - 1
    right = idx
    choose_right = (timestamps[right] - target_times) < (target_times - timestamps[left])
    return np.where(choose_right, right, left)


def extract_pos_vel_acc(raw_path):
    '''
    Extract 2D poses, velocities in vehicle frame, and IMU accelerations from Novatel data. Output data in 2Hz.
    Args:
        raw_path (Path): Path to the raw data directory.
    Returns:
        dict: A dictionary containing:
            poses (np.ndarray): Nx4 array of 2D poses (x, y, yaw).
            vel (np.ndarray): Nx2 array of velocities in vehicle frame (v_x, v_y). 
            acc (np.ndarray): Nx2 array of accelerations in vehicle frame (a_x, a_y).
            times (np.ndarray): N array of timestamps for poses and velocities.
            imu_times (np.ndarray): N array of timestamps for IMU data. 
            start_time_pose (float): Start time for pose data.
    '''
    
    import convert_novatel_to_pose

    novatel_data_path = raw_path / "raw" / "novatel" / "data"
    novatel_ts_path = raw_path / "raw" / "novatel" / "timestamps.txt"
    novatel_imu_data_path = raw_path / "raw" / "novatel_imu" / "data"
    novatel_imu_ts_path = raw_path / "raw" / "novatel_imu" / "timestamps.txt"

    # Data format:
    # latitude (degrees)
    # longitude (degrees)
    # altitude (m)
    # undulation (m)
    # latitude_std (m)
    # longitude_std (m)
    # altitude_std (m)
    # roll (degrees)
    # pitch (degrees)
    # azimuth (degrees)
    # roll_std (degrees)
    # pitch_std (degrees)
    # azimuth_std (degrees)
    # ins_status
    # position_type
    # extended_status
    # seconds_since_update (seconds)
    # north_velocity (m/s)
    # east_velocity (m/s)
    # up_velocity (m/s)
    # north_velocity_std (m/s)
    # east_velocity_std (m/s)
    # up_velocity_std (m/s)
    novatel_data = load_data(novatel_data_path)
    poses = convert_novatel_to_pose.convert_novatel_to_pose(novatel_data)
    times, start_time = load_timestamps(novatel_ts_path) 
    # This is at 20 Hz. Downsample to 2 Hz by taking every 10th frame
    poses_downsampled = poses[::10]
    times_downsampled = times[::10]
    poses_2d = convert_to_2d(poses_downsampled) # Convert to 2D
    yaw = np.deg2rad(90 - novatel_data[::10, 9]) # Azimuth in radians

    # Convert world -> vehicle (x forward, y left): rotate by -yaw
    v_n = novatel_data[::10, 17]                    # North
    v_e = novatel_data[::10, 18]                    # East
    v_x =  v_e * np.cos(yaw) + v_n * np.sin(yaw)
    v_y = -v_e * np.sin(yaw) + v_n * np.cos(yaw)
    velocities_vehicle_frame = np.stack([v_x, v_y], axis=1)

    # Novatel IMU data
    # Format:
    # pitch_rate (rad/sample)
    # roll_rate (rad/sample)
    # yaw_rate (rad/sample)
    # x_accel (m/s/sample)
    # y_accel (m/s/sample)
    # z_accel (m/s/sample)
    novatel_imu_data = load_data(novatel_imu_data_path)
    novatel_imu_times, novatel_imu_start_time = load_timestamps(novatel_imu_ts_path)
    novatel_imu_times_aligned = novatel_imu_times + (np.timedelta64(novatel_imu_start_time - start_time) / np.timedelta64(1, 's')) # align start times
    # Find corresponding indices in novatel_imu_times for times_downsampled
    novatel_imu_indices = find_closest_timestamp_inds(times_downsampled, novatel_imu_times_aligned)

    acc_downsampled = novatel_imu_data[novatel_imu_indices, 3:5] * 100  # Convert from per sample to per second (100 Hz)
    acc_times_downsampled = novatel_imu_times_aligned[novatel_imu_indices]

    # Convert to vehicle frame
    acc_lat = acc_downsampled[:, 0] # right
    acc_lon = acc_downsampled[:, 1] # forward

    acc_x = acc_lon
    acc_y = -acc_lat

    # Return dictionary
    return {
        "poses": poses_2d,
        "velocities": velocities_vehicle_frame,
        "accelerations": np.stack([acc_x, acc_y], axis=1),
        "pose_times": times_downsampled,
        "imu_times": acc_times_downsampled,
        "start_time_pose": start_time,
    }