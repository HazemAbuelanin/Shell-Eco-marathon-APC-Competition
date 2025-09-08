import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import time
import tf2_ros
import tf2_geometry_msgs
import tf
from geometry_msgs.msg import TransformStamped

def manual_ransac(points, distance_threshold=0.2, iterations=100):
    best_inliers = []
    if len(points) < 3:
        return np.array([]), points.copy()
    for _ in range(iterations):
        # Use basic numpy operations for better compatibility
        sample_indices = np.random.permutation(len(points))[:3]
        sample_points = points[sample_indices]
        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal = normal / norm  # More explicit division for older numpy
        d = -np.dot(normal, sample_points[0])
        distances = np.abs(np.dot(points, normal) + d)  # More explicit dot product
        inliers = np.where(distances < distance_threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    ground = points[best_inliers]
    obstacles = np.delete(points, best_inliers, axis=0)
    return ground, obstacles

def manual_voxel_filter(points, voxel_size=0.1):
    if len(points) == 0:
        return points
    # Use basic numpy operations for better compatibility
    voxel_coords = np.floor(points / voxel_size)
    unique_voxels, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    voxel_centroids = np.zeros((len(unique_voxels), 3))
    for i in range(len(unique_voxels)):
        mask = (inverse == i)
        voxel_points = points[mask]
        voxel_centroids[i] = np.mean(voxel_points, axis=0)
    return voxel_centroids

def compute_pca_obb(cluster_points, template_dims=None):
    """
    Computes a PCA-aligned OBB with real spread along XY (PCA) and fixed Z.

    Args:
        cluster_points: (N, 3) point cloud in LiDAR frame
        template_dims: optional [L, W, H], passed through for external use

    Returns:
        dict with center, extents, rotation matrix, quaternion, corners, and template
    """
    if len(cluster_points) < 3:
        return None

    # Mean and centering
    mean = np.mean(cluster_points, axis=0)
    centered = cluster_points - mean

    # 2D PCA in XY plane
    xy_centered = centered[:, :2]
    cov_xy = np.cov(xy_centered.T)
    vals_xy, vecs_xy = np.linalg.eigh(cov_xy)
    xy_axes = vecs_xy[:, np.argsort(vals_xy)[::-1]]  # (2, 2)

    # Form candidate 3D axes
    x_axis = np.array([xy_axes[0, 0], xy_axes[1, 0], 0.0])
    y_axis = np.array([xy_axes[0, 1], xy_axes[1, 1], 0.0])

    # Normalize
    x_axis = x_axis / np.linalg.norm(x_axis)  # More explicit division
    y_axis = y_axis / np.linalg.norm(y_axis)  # More explicit division

    # Project original points into PCA frame
    rot_2d = np.vstack([x_axis[:2], y_axis[:2]]).T  # More explicit stacking
    projected_2d = np.dot(xy_centered, rot_2d)  # More explicit dot product
    spread = np.max(projected_2d, axis=0) - np.min(projected_2d, axis=0)

    # Swap axes if spread in y direction is greater (i.e., width > length)
    if spread[1] > spread[0]:
        x_axis, y_axis = y_axis, x_axis
        spread = spread[::-1]

    # Flip x_axis to ensure it's pointing forward (+X in LiDAR frame)
    if x_axis[0] < 0:
        x_axis = -x_axis
        y_axis = -y_axis

    # Create full rotation matrix
    z_axis = np.array([0, 0, 1])
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)  # re-orthogonalize
    x_axis = x_axis / np.linalg.norm(x_axis)  # More explicit division
    y_axis = y_axis / np.linalg.norm(y_axis)  # More explicit division
    axes = np.vstack([x_axis, y_axis, z_axis]).T  # More explicit stacking

    # Project into PCA-aligned 3D frame
    proj = np.dot(centered, axes)  # More explicit dot product
    min_proj = np.min(proj, axis=0)
    max_proj = np.max(proj, axis=0)
    extents = max_proj - min_proj
    center_pca = 0.5 * (min_proj + max_proj)

    # Backproject center to world frame
    obb_center = np.dot(axes, center_pca) + mean  # More explicit dot product

    # Apply template correction (if any)
    if template_dims is not None:
        if 1.3 > obb_center[1] > -1.3:
            if 1 < extents[0] <= 0.8 * template_dims[0] or 1 < extents[1] <= 0.8 * template_dims[1]:
                offset = (template_dims[0] - extents[0]) / 2
                if obb_center[0] > 0:
                    obb_center[0] += offset
                else:
                    obb_center[0] -= offset
                
                extents[1] = template_dims[1]
                x_axis, y_axis = y_axis, x_axis
                extents[0] = template_dims[0]

        else:
            if extents[0] < 1 or extents[0] > 4.75 or extents[1] < 1 or extents[1] > 1.85:
                return None

    

    # Quaternion
    rotation_quat = R.from_matrix(axes).as_quat()

    return {
        'center':      obb_center,
        'extents':     extents,
        'rotation':    axes,
        'quaternion':  rotation_quat.tolist(),
    }

def lookup_latest_transform_with_wait(tf_listener, target_frame, source_frame, timeout_sec=0.5):
    """
    ROS 1 version of transform lookup with waiting.
    
    Args:
        tf_listener: tf.TransformListener instance
        target_frame: target frame name
        source_frame: source frame name
        timeout_sec: timeout in seconds
        
    Returns:
        TransformStamped message
    """
    start_time = rospy.Time.now()
    timeout = rospy.Duration(timeout_sec)
    
    while not rospy.is_shutdown():
        try:
            # Use waitForTransform and lookupTransform from tf.TransformListener
            tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(0), timeout)
            transform = tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            
            # Convert tf transform to TransformStamped
            transform_stamped = TransformStamped()
            transform_stamped.header.frame_id = target_frame
            transform_stamped.child_frame_id = source_frame
            transform_stamped.header.stamp = rospy.Time.now()
            transform_stamped.transform.translation.x = transform[0][0]
            transform_stamped.transform.translation.y = transform[0][1]
            transform_stamped.transform.translation.z = transform[0][2]
            transform_stamped.transform.rotation.x = transform[1][0]
            transform_stamped.transform.rotation.y = transform[1][1]
            transform_stamped.transform.rotation.z = transform[1][2]
            transform_stamped.transform.rotation.w = transform[1][3]
            
            return transform_stamped
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            if (rospy.Time.now() - start_time) > timeout:
                raise e
            rospy.sleep(0.01)
