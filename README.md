# **Exploring AlexNet for ImageNet Classification**

**Unleashing Vision, One Image at a Time**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xxJVp0L6WoaWkKi4cW18-oTo3OCyEpPM?usp=sharing)

This github repository demonstrates the implementation of **AlexNet** for image classification on the ImageNet dataset. By leveraging **transfer learning** and **fine-tuning** techniques, it achieves high accuracy and robust classification performance. The project consolidates key results, metrics, and visualizations into a single **Jupyter/Colab notebook** notebook, making it accessible and easy to use.

---

## **Features**
- **Model**: AlexNet architecture adapted for 10 ImageNet classes.
- **Performance**: Achieved **93.25% accuracy** with strong precision, recall, and F1-scores.
- **Visualization**: Includes Confusion matrix, Learning curves, Metric breakdowns.
- **Optimization**: Leveraged transfer learning, fine-tuning, and a one-cycle learning policy for efficient training.

---

## **Usage**

To explore and run this project, follow these steps:

1. Open the Colab notebook using the link below:
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xxJVp0L6WoaWkKi4cW18-oTo3OCyEpPM?usp=sharing)

2. Execute the cells sequentially to:
   - Load and preprocess the ImageNet dataset.
   - Train the AlexNet model using transfer learning and fine-tuning.
   - Visualize the results, including performance metrics and training progress.

3. Download the project documentation for in-depth insights and methodology:
   [Project Documentation]([https://drive.google.com/drive/folders/YOUR_DOCUMENTATION_LINK](https://drive.google.com/file/d/1CEXEDjnSd4ZIbAAucRzi2b82pgHqU06a/view?usp=sharing))
---

## **Results Summary**

- **Accuracy**: Achieved **93.25%** across 10 ImageNet classes.
- **Metrics**: High precision, recall, and F1-scores across all classes, indicating robust classification performance.
- **Learning Curve**: Consistent training loss reduction and early convergence of validation accuracy.

### **Visualizations**
| Metric      | Value      |
|-------------|------------|
| Accuracy    | 93.25%     |
| Precision   | 93.50%     |
| Recall      | 93.24%     |
| F1-Score    | 93.31%     |

Below are key visualizations from the project:

#### **1. ImageNet Dataset Sample**
![ImageNet Dataset](images/imagenet_visualization.png)

#### **2. Training Loss Over Epochs**
![Training Loss](images/training_loss.png)

#### **3. Validation Accuracy Over Epochs**
![Validation Accuracy](images/validation_accuracy.png)

#### **4. Learning Curve**
![Learning Curve](images/learning_curve.png)

#### **5. Overall Metrics**
![Overall Metrics](images/overall_metrics.png)

#### **6. Confusion Matrix**
![Confusion Matrix](images/confusion_matrix.png)

#### **7. Model Performance Visualization**
![Model Performance](images/model_performance.png)

#### **8. Model Architecture**
![Model Architecture](images/model_architecture.png)

---

## **Insights**

- **Transfer Learning**: Utilized pre-trained AlexNet weights, reducing training time and improving accuracy.
- **Fine-Tuning**: Adaptive learning rates enhanced performance on the target dataset.
- **Efficiency**: Combined transfer learning with the one-cycle learning policy for faster convergence and minimized overfitting.

---

## **Requirements**

To run the notebook, ensure you have the following installed in your environment (if running locally):
- **Python**: Version 3.8 or later
- **Libraries**: 
  - PyTorch
  - torchvision
  - NumPy
  - Matplotlib
  - pandas
  - scikit-learn

To install the required libraries, execute:
```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn
