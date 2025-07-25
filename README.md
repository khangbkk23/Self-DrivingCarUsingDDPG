# Self-DrivingCarUsingDDPG
##  Self-Driving Car with DDPG and Perception Integration

This is a personal self-driving car project that integrates **Deep Reinforcement Learning (DDPG)** for decision-making and **pretrained models** (e.g. YOLOv8, SegFormer) for perception.

The goal is to train an autonomous driving agent capable of making safe driving decisions (e.g. lane-keeping, obstacle avoidance) by using perception input and reinforcement learning techniques.

---

## 🚗 Project Overview

- **Perception module** uses pretrained models (e.g. YOLOv8) to detect objects and extract scene understanding.
- **DDPG agent** learns continuous control actions in simulation (e.g. CARLA) based on the perception inputs.
- **Continual Learning** methods are optionally integrated to improve robustness in changing environments (weather, maps...).

---

## 📦 Project Structure


