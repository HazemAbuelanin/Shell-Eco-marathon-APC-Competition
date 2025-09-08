# The Software Stack That Achieved Global Second Place at the Shell Eco-marathon APC 2025

This repository contains the complete software system developed for the **Shell Eco-marathon Autonomous Programming Competition 2025**, where it secured **global second place**. The system was implemented and tested on the **CARLA Simulator**, integrating classical perception, planning, and control methods with efficiency-optimized behavior planning.  

---

## 🏆 Competition Objective  
The challenge was to autonomously navigate through **14 target waypoints** while:  
- Minimizing **energy consumption**.  
- Ensuring **collision-free** navigation.  
- Maintaining **safe and legal driving behavior**.  

![Alt text](Shell Paper.drawio)
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

##  Key Highlights  
- Achieved **2nd place globally** with this stack.  
- Strong balance of **classical perception algorithms** and **lightweight planning/control** methods.  
- Focused on **energy efficiency** while preserving safety.  
[See results](https://www.shellecomarathon.com/2025-programme/autonomous-programming-competition.html)

---

## ⚙️ Build and Run Instructions (Catkin Tools)  

This repository is designed to be built within a **ROS Catkin workspace**. Place the repo inside `src/` of your catkin workspace, then follow these steps:  

### 1. Initialize catkin workspace  
```bash
catkin init
```

### 2. Configure to use install space  
```bash
catkin config --install
```

### 3. Set build options  
```bash
catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSETUPTOOLS_DEB_LAYOUT=OFF
```

### 4. Install dependencies (from workspace root)  
```bash
rosdep update
rosdep install . -q -y --from-paths -i
```

### 5. Build the workspace  
```bash
catkin build
```

### 6. Source installed environment  
```bash
source install/setup.bash
```

### 7. Run the main launch file  
```bash
roslaunch shell_simulation shell_simulation.launch
```

---

## 🛠 Alternative Build (catkin_make)  

If you are using **catkin_make**:  

```bash
catkin_make
source devel/setup.bash
roslaunch shell_simulation shell_simulation.launch
```

---

## 🔗 External Dependencies  

Before running, ensure you have the Shell Eco-marathon CARLA + ROS bridge docker environment installed:  

- [SEM-APC Student Docker Environment](https://github.com/swri-robotics/sem-apc-student-docker-environment)  
- [SEM-APC Example Project](https://github.com/swri-robotics/sem-apc-example-project)  

### Requirements  
- **Ubuntu 22.04 (preferred)**  
- Docker environment above installed and running  
- `carla_config.yaml` must be edited to match the one provided in this repository  

### Launch Order  
1. Start the **simulation environment** (CARLA + shell ROS bridge):  
```bash
roslaunch carla_shell_bridge main.launch
```  

2. Run this software stack:  
```bash
roslaunch shell_simulation shell_simulation.launch
```  

---

## 📌 Notes  
- Ensure that `carla_config.yaml` matches the configuration in this repo before starting.  
- Always launch the CARLA bridge before starting simulation nodes.  
