# PFSSL
[![AAAI](https://img.shields.io/badge/AAAI25-Paper-blue)]()
Official code for AAAI 2025 paper: Dual-calibrated Co-training Framework for Personalized Federated Semi-Supervised Medical Image Segmentation
![](./unrelated/poster.jpg)

## Abstract
Federated Semi-Supervised Learning (FSSL) has emerged as a crucial topic in medical image analysis, allowing multiple medical institutions to collaboratively train a global model using limited labeled data. However, existing FSSL methods focus solely on an effective combination of federated learning and semi-supervised learning, ignoring the heterogeneity of client data and the inadaptability of semi-supervised methods in diverse environments, which leads to knowledge bias in local models and impedes stable convergence. To this end, we explore the application of personalization in FSSL and propose a novel dual-calibrated co-training framework. To adapt to the unique feature distribution of client data, we consider collaborative relationships among clients to aggregate a personalized model for each client. We further build a dualstudent architecture with the personalized model and private local model on the client side, which encourages model disagreement for co-training while enhancing participant privacy. Most importantly, we design dual calibration strategies that adaptively optimize the model: Local calibration improves the boundary discrimination of the local model by dynamically replacing pseudo-label boundary patches; Global calibration corrects model direction based on the real-time perception of the biases between local dual-student models. Experimental results show the effectiveness of our method on a private medical dataset and two public medical datasets.
