# 🚦 Multi-Task Traffic Sign Recognition System

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview

This project implements a production-ready **Multi-Head Convolutional Neural Network (CNN)** capable of performing three simultaneous classification tasks from a single traffic sign image:
1.  **Sign Type Classification** (43 Classes)
2.  **Shape Detection** (5 Classes)
3.  **Color Recognition** (4 Classes)

Unlike standard classifiers, this system leverages **Multi-Task Learning (MTL)** to share feature extraction layers across tasks, improving efficiency and reducing inference latency. The model is deployed as a user-friendly web application using **Streamlit**.

---

## 🚀 Key Features

* **Multi-Head Architecture:** A single backbone feeds into three distinct dense heads for independent outputs.
* **End-to-End Pipeline:** Preprocessing (Rescaling, Resizing) is embedded directly within the TensorFlow computation graph, ensuring zero training-serving skew.
* **Production Stack:** Built on **TensorFlow 2.19** and **Python 3.12**, utilizing the modern `.keras` serialization format.
* **Interactive Dashboard:** Real-time inference visualization with probability distributions and confidence metrics.

---

## 🛠️ Architecture & Engineering

### Model Topology
The model accepts raw RGB images of shape `(N, H, W, 3)` and processes them through a shared feature extractor before branching out.

```mermaid
graph TD
    A[Input Image] --> B[Shared CNN Backbone]
    B --> C{Feature Flattening}
    C --> D[Head 1: Sign Type]
    C --> E[Head 2: Shape]
    C --> F[Head 3: Color]
    D --> O1[Output: Softmax (43)]
    E --> O2[Output: Softmax (5)]
    F --> O3[Output: Softmax (4)]