# F1TENTH Autonomous Racing Project

Reinforcement learning based autonomous racing research using the F1TENTH platform.

- **Period:** Fall 2025 – Spring 2026
- **Affiliation:** Department of Mechanical Engineering, Chonnam National University
- **Team:** Echo
- **Team Leader:** Sangrok So (소상록)
- **Members:** Jeongdo Choi (최정도), Hyunwoo Kang (강현우)
- **Advisor:** Prof. S. Ko

[강화학습서킷주행최적화.pdf](https://github.com/user-attachments/files/32135854/default.pdf)

---

## Research Overview

This project investigates reinforcement learning based autonomous racing
without prior mapping and path planning.



---

## 1. Limitation of Model-Based Control
<img width="1920" height="1080" alt="슬라이드3" src="https://github.com/user-attachments/assets/96d29c22-2d44-4b5b-af58-eb761a0fef71" />

<img width="1920" height="1080" alt="슬라이드4" src="https://github.com/user-attachments/assets/e6b3e91f-dcab-448e-b077-4adf70cb8a53" />

<img width="1920" height="1080" alt="슬라이드5" src="https://github.com/user-attachments/assets/5f0b5608-cec7-4a43-aec9-76ed7a12bf7a" />

![Uploading 슬라이드7.PNG…]()
<img width="1920" height="1080" alt="슬라이드6" src="https://github.com/user-attachments/assets/df92ec15-f3f9-4c76-af2b-feb9ee9477d7" />

The previous model-based approach required mapping, localization,
and path planning before autonomous driving.

---

## 2. Reinforcement Learning Approach

<img width="1920" height="1080" alt="슬라이드8" src="https://github.com/user-attachments/assets/8044ba19-c62b-4412-aba7-5f78863bb833" />
<img width="1920" height="1080" alt="슬라이드9" src="https://github.com/user-attachments/assets/7059a65a-d8bc-4ad5-b992-6e0a68e2a73b" />


Raw LiDAR scan data is directly processed by the RL agent
to generate steering and velocity commands.

---

## 3. RL Algorithm Comparison

<img width="1920" height="1080" alt="슬라이드10" src="https://github.com/user-attachments/assets/5f5e882e-82ce-405d-9bd4-cfc8000cb5ab" />
<img width="1920" height="1080" alt="슬라이드11" src="https://github.com/user-attachments/assets/4b7cc156-dc4d-4e68-b08c-b863ad92997a" />

<img width="1920" height="1080" alt="슬라이드12" src="https://github.com/user-attachments/assets/e19059e7-4f56-4dcf-a69b-f7d0fdad3fd6" />


DDPG, SAC, and TD3 were evaluated.
TD3 was selected and further improved through:

- Larger neural network
- Increased batch size
- Noise decay

---

## 4. Simulation Results

<img width="1920" height="1080" alt="슬라이드13" src="https://github.com/user-attachments/assets/a3ab8a96-94b5-427f-8c7d-08fb803fdb36" />
<img width="1920" height="1080" alt="슬라이드14" src="https://github.com/user-attachments/assets/92309bfb-2fb7-4d9a-9da8-019b4ee97664" />
<img width="1920" height="1080" alt="슬라이드15" src="https://github.com/user-attachments/assets/028ecca6-383b-4a7d-9213-641e61c5dc0a" />




TD3(Ours) achieved the highest overall weighted driving score.

**Driving Score: 0.917**

---

## 5. Sim-to-Real

### LiDAR Issue

<img width="1920" height="1080" alt="슬라이드17" src="https://github.com/user-attachments/assets/bed926ca-d0aa-4cf6-a780-d31ae46b4762" />


LiDAR ground-detection errors caused by vehicle pitch were identified
and corrected using a redesigned 3D-printed sensor mount.

### Control Stabilization

<img width="1920" height="1080" alt="슬라이드18" src="https://github.com/user-attachments/assets/2fb2ca24-b29d-4927-bb1c-0573ce6e2a49" />


Steering smoothing and distance-based velocity constraints were
implemented for stable real-world driving.

---

## 6. Real Vehicle Test

<img width="1920" height="1080" alt="슬라이드19" src="https://github.com/user-attachments/assets/0de95de6-a12f-433a-abac-740b5555b048" />



**40 consecutive laps without collision**

**Best Lap Time: 16.8 s**

---

## 7. Results
<img width="1920" height="1080" alt="슬라이드20" src="https://github.com/user-attachments/assets/80fea0ca-7018-4e14-ad6d-01cd93cec87b" />

<img width="1920" height="1080" alt="슬라이드22" src="https://github.com/user-attachments/assets/ac1dde2e-df75-4a9d-b206-5e501f7c1b43" />

<img width="1920" height="1080" alt="슬라이드23" src="https://github.com/user-attachments/assets/1bf15d4d-c42f-4c76-be2f-32ca4c01d032" />





| Metric | Result |
|---|---:|
| RL Driving Score | 0.917 |
| Simulation Lap Time | 10.70 s |
| Sim-to-Real Lap Time | 16.8 s |
| Continuous Driving | 40 Laps |
| Simulation Lap Time Improvement | 31.8% |
| Real Vehicle Lap Time Improvement | 58.4% |[강화학습서킷주행최적화.pdf](https://github.com/user-attachments/files/32089091/default.pdf)
