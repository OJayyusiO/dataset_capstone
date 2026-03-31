# Capstone Project: Autonomous Vision System

This repository contains the source code for the Vision Team's capstone. 

> **IMPORTANT:** The CARLA Simulator (approx. 20GB) is excluded from this repository via `.gitignore`. You must download and place it manually following the steps below.

---

## 🛠 Setup Procedure

### 1. Repository & Simulator
1. Clone this repository to your local machine.
2. Download the **CARLA 0.9.16 Windows** zip from [CARLA Releases](https://github.com/carla-simulator/carla/releases/tag/0.9.16/).
3. Extract it directly into the root of this project. 

**Your folder structure MUST look like this for the scripts to find the API:**
``` 
capstone_sim/
├── Code/
├── .gitignore
├── environment.yml
└── CARLA_0.9.16/             <-- The extracted folder
    ├── CarlaUE4.exe          <-- The simulator app
    └── PythonAPI/            <-- The library files