# The Software Stack That Achieved Global Second Place at the Shell Eco-marathon APC

This repository contains the complete software system developed for the **Shell Eco-marathon Autonomous Programming Competition 2025**, where it secured **global second place**. The system was implemented and tested on the **CARLA Simulator**, integrating classical perception, planning, and control methods with efficiency-optimized behavior planning.  

---

## 🏆 Competition Objective  
The challenge was to autonomously navigate through **14 target waypoints** while:  
- Minimizing **energy consumption**.  
- Ensuring **collision-free** navigation.  
- Maintaining **safe and legal driving behavior**.  

Our approach combined **dynamic programming for optimal waypoint traversal**, a robust perception stack with **LiDAR–camera fusion**, and an **adaptive planning and control framework** designed for energy efficiency.  

---

## 📂 Project Structure  

```
project/
│
├── custom_msg/                # Custom ROS message definitions
│   ├── msg/                   # Object info, dimensions, traffic lights, perception outputs
│   ├── include/custom_msg/    # Message headers
│   ├── src/                   # Supporting C++ code (if any)
│   ├── package.xml
│   └── CMakeLists.txt
│
├── shell_simulation/          # Core simulation and stack implementation
│   ├── src/shell_simulation/  # Core Python modules
│   │   ├── kalman_filter.py   # Multi-object tracking with Kalman filter + Hungarian algorithm
│   │   ├── lidar_utilities.py # Classical 3D LiDAR preprocessing, segmentation, clustering
│   │   ├── controller.py      # Lateral (Pure Pursuit) & Longitudinal (MPC/Adaptive PID) control
│   │   └── graph.py           # Utilities for TSP (Dynamic Programming)
│   │
│   ├── data/                  # Precomputed CSV files
│   │   ├── trajectory.csv     # Global path waypoints
│   │   ├── centroids.csv      # Clustered LiDAR centroids
│   │   └── traffic_lights_info.csv # Traffic light locations & IDs
│   │
│   ├── scripts/               # ROS Nodes
│   │   ├── lidar_node.py          # LiDAR-based 3D object detection
│   │   ├── planner_perception_node.py # Fusion of LiDAR & camera detections
│   │   ├── behavior_planner_node.py   # Mode selection (stop, proceed, follow, maneuver)
│   │   ├── path_tracker_node.py       # Path tracking & Frenet planner integration
│   │   ├── controller_node.py         # Interface to low-level control
│   │   └── logger_node.py             # Logging system states & results
│   │
│   ├── launch/                # ROS launch files
│   │   ├── shell_simulation.launch
│   │   ├── path_tracker_controller.launch
│   │   └── test.launch
│   │
│   ├── resource/              # ROS package resource markers
│   ├── package.xml
│   ├── setup.py
│   └── CMakeLists.txt
│
└── ...
```

---

## 🔑 System Overview  

### 1. Global Planning  
- **Traveling Salesman Problem (TSP):** Solved with **Dynamic Programming**, ensuring the most energy-efficient order of visiting all 14 waypoints.  
- Planned path stored in **trajectory.csv** for downstream modules.  

### 2. Perception  
- **3D LiDAR (Velodyne VLP-16):**  
  - Preprocessing: voxel grid downsampling, noise filtering, ground plane removal.  
  - Segmentation: Euclidean clustering/DBSCAN.  
  - Feature extraction: geometric + statistical features per object.  
- **Camera (RGB-D):** 2D object detection (YOLOv11 Nano).  
- **Traffic Lights:** YOLO detection + OpenCV color classification.  
- **Sensor Fusion:** Combines LiDAR 3D geometry with camera semantic classification.  
- **Multi-object Tracking:** Kalman Filter + Hungarian algorithm for consistent IDs.  

### 3. Behavior Planning  
Evaluates ego state and perception to select **driving mode**:  
- **Stop:** When red light or lead vehicle stops (outputs stopping distance).  
- **Proceed:** Road is clear → follow global path.  
- **Follow:** Adaptive Cruise Control (ACC) with relative speed + distance.  
- **Maneuver:** Safe overtaking using Frenet planner.  

The **Frenet planner** is activated only during overtakes to save computation, optimizing both **safety cost** and **energy cost**.  

### 4. Control  
- **Lateral Control:** Adaptive Pure Pursuit with variable lookahead.  
- **Longitudinal Control:** MPC or Adaptive PID for smooth, energy-efficient speed regulation.  

---

## 🚀 Key Highlights  
- Achieved **2nd place globally** with this stack.  
- Strong balance of **classical perception algorithms** and **lightweight planning/control** methods.  
- Focused on **energy efficiency** while preserving safety.  
