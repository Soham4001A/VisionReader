# VAE-Augmented Vision Pipeline: Real-Time Hand Pose Classification

  <!-- It is highly recommended to create a GIF of the final demo and replace this link

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Library-OpenCV-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/Tool-MediaPipe-red.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
</p> -->

This project demonstrates a powerful strategy for building robust computer vision models in **data-stricken environments**. I've created a full-stack pipeline that tacklesto demonstrate a sample solution to a common challenge: the lack of a large, labeled dataset.

The core of this project is the use of a **Variational Autoencoder (VAE)** to generate thousands of synthetic, high-quality hand images. This synthetic dataset is then used to train a lightweight classifier for a real-world task: **real-time finger counting from a live webcam feed**.

This repository is a showcase of how to bootstrap a complex computer vision system from a small initial dataset, a technique applicable to countless target recognition and classification problems.

---

## 🎯 The Core Idea: From Data Scarcity to Abundance

Many real-world machine learning projects fail due to a lack of labeled training data. This project directly confronts that issue with a hybrid learning approach.

Follow the workflow-

 <!-- It's recommended to create a simple flowchart image for this -->

1.  **Collect Small "Real" Dataset:** Start by capturing a few hundred images of a subject (in this case, my hand). This takes only a few minutes.
2.  **Train an Unsupervised Model (VAE):** The VAE learns a compressed, continuous *latent space* representation of what a hand looks like. It doesn't need any labels; it just learns to reconstruct the input images.
3.  **Generate a Large Synthetic Dataset:** By sampling from the VAE's latent space, I can generate thousands of novel hand images with variations in pose, orientation, and lighting that didn't exist in my original small dataset.
4.  **Automated Labeling:** I use Google's MediaPipe framework as an "oracle" to automatically and accurately label my entire dataset (both real and synthetic) with the correct finger count. This step alone saves hundreds of hours of manual labor.
5.  **Train a Supervised Classifier:** With a large, labeled dataset now available, I train an efficient Convolutional Neural Network (ResNet18) to perform the final classification task.
6.  **Deploy in Real-Time:** The final, lightweight model is deployed in a real-time OpenCV pipeline that detects hands, classifies the finger count, and displays the results on a live webcam feed.

---

## ✨ Key Features

*   **VAE for Data Augmentation:** A PyTorch-based Variational Autoencoder that generates thousands of synthetic hand images from a small seed dataset.
*   **Automated Data Labeling:** A clever pipeline that uses MediaPipe Hands to programmatically label over 10,000 images, creating a high-quality training set with zero manual effort.
*   **High-Performance Classifier:** A fine-tuned ResNet18 model that achieves high accuracy on the finger counting task (classes 0-5).
*   **Real-Time Inference:** An efficient end-to-end pipeline using OpenCV for video capture and MediaPipe for hand localization, running smoothly on a standard laptop.
*   **Modular & Reproducible:** The entire project is broken down into numbered Python scripts, allowing anyone to easily reproduce the results step-by-step.

---

## 🛠️ Tech Stack

| Component | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **ML Framework** | PyTorch | For building and training the VAE and Classifier models. |
| **Hand Detection** | Google MediaPipe | For high-performance hand landmark detection, used in both labeling and the final demo. |
| **Computer Vision** | OpenCV | For all image/video processing, data collection, and rendering the final output. |
| **Data Handling** | NumPy, Pillow | For numerical operations and image manipulation. |
| **Project Setup**| `venv`, `pip` | For managing a clean and reproducible environment. |

---

## 🚀 Getting Started

You can replicate this entire project on your local machine. The scripts are numbered to be run in sequence.

### Step 0: Prerequisites

First, clone the repository to your local machine.

```bash
git clone https://github.com/your-username/vae-hand-pose-classifier.git
cd vae-hand-pose-classifier
```

### Step 1: Set Up the Environment

The `setup.sh` script creates a Python virtual environment and installs all necessary dependencies.

```bash
bash setup.sh

# Activate the environment for your terminal session
source venv/bin/activate
```

### Step 2: Collect Your Initial Data

Run the first script to collect a small set of "real" hand images using your webcam. The more variety in poses and lighting, the better!

```bash
python 1_data_collector.py
```
*   Follow the on-screen instructions. Press `s` to save an image and `q` to quit.
*   Your images will be saved in the `raw_data/` directory.

### Step 3: Train the VAE

Train the Variational Autoencoder on your collected images. This will learn the "essence" of a hand.

```bash
python 2_vae_trainer.py
```
*   This will generate a `models/vae.pth` file.

### Step 4: Generate Thousands of Synthetic Images

Use the trained VAE to generate a large synthetic dataset.

```bash
python 3_generate_synthetic_data.py
```
*   This populates the `synthetic_data/` directory with 5,000 new images.

### Step 5: Run the Auto-Labeler

This is the key step where all our data gets processed and labeled. The script uses MediaPipe to determine the finger count for every image in `raw_data/` and `synthetic_data/`.

```bash
python 4_label_data.py
```
*   The script will create a fully labeled and organized dataset in `processed_data/`, split into `train` and `val` sets.

### Step 6: Train the Final Classifier

Now, train the ResNet18 classifier on your rich, combined dataset.

```bash
python 5_classifier_trainer.py
```
*   This will generate the final `models/finger_classifier.pth` file.

### Step 7: Run the Live Demo!

Execute the final script to see your model in action!

```bash
python 6_real_time_demo.py
```
*   Point your webcam at your hand and watch the model classify the number of fingers in real-time. Press `q` to quit.

---

## 🔮 Future Work & Extensions

This project architecture is highly extensible. Here are some ideas for taking it to the next level:

*   **Sign Language Recognition:** Extend the classifier to recognize a wider range of static hand poses corresponding to letters in American Sign Language (ASL).
*   **Dynamic Gesture Recognition:** Implement an LSTM or Transformer model on top of the CNN to understand sequences of gestures, enabling basic sign language translation.
*   **Deploy to the Edge:** Optimize the model using ONNX or TensorFlow Lite and deploy it on an edge device like a Raspberry Pi or Jetson Nano for a standalone accessibility tool.
*   **GANs vs. VAEs:** Swap out the VAE for a Generative Adversarial Network (GAN) to compare the quality and diversity of the generated synthetic data.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
