# F1TENTH Autonomous Racing Project

Reinforcement learning based autonomous racing research using the F1TENTH platform.

- **Period:** Fall 2025 – Spring 2026
- **Affiliation:** Department of Mechanical Engineering, Chonnam National University
- **Team:** Echo
- **Team Leader:** Sangrok So (소상록)
- **Members:** Jeongdo Choi (최정도), Hyunwoo Kang (강현우)
- **Advisor:** Prof. S. Ko

---

## Research Overview

This project investigates reinforcement learning based autonomous racing
without prior mapping and path planning.

📄 [Full Research Presentation](docs/F1TENTH_RL_Research_Presentation_2026.pdf)

---

## 1. Limitation of Model-Based Control

![Model-Based Limitation](docs/slides/03_model_based_limitation.png)

The previous model-based approach required mapping, localization,
and path planning before autonomous driving.

---

## 2. Reinforcement Learning Approach

![RL Goal](docs/slides/04_rl_goal.png)

Raw LiDAR scan data is directly processed by the RL agent
to generate steering and velocity commands.

---

## 3. RL Algorithm Comparison

![TD3 Improvement](docs/slides/05_td3_improvement.png)

DDPG, SAC, and TD3 were evaluated.
TD3 was selected and further improved through:

- Larger neural network
- Increased batch size
- Noise decay

---

## 4. Simulation Results

![Simulation Results](docs/slides/06_simulation_results.png)

TD3(Ours) achieved the highest overall weighted driving score.

**Driving Score: 0.917**

---

## 5. Sim-to-Real

### LiDAR Issue

![LiDAR Issue](docs/slides/07_lidar_issue.png)

LiDAR ground-detection errors caused by vehicle pitch were identified
and corrected using a redesigned 3D-printed sensor mount.

### Control Stabilization

![Control Issue](docs/slides/08_control_issue.png)

Steering smoothing and distance-based velocity constraints were
implemented for stable real-world driving.

---

## 6. Real Vehicle Test

![Sim-to-Real](docs/slides/09_sim_to_real.png)

**40 consecutive laps without collision**

**Best Lap Time: 16.8 s**

---

## 7. Results

![Final Results](docs/slides/10_final_results.png)

| Metric | Result |
|---|---:|
| RL Driving Score | 0.917 |
| Simulation Lap Time | 10.70 s |
| Sim-to-Real Lap Time | 16.8 s |
| Continuous Driving | 40 Laps |
| Simulation Lap Time Improvement | 31.8% |
| Real Vehicle Lap Time Improvement | 58.4% |[강화학습서킷주행최적화.pdf](https://github.com/user-attachments/files/32089091/default.pdf)
