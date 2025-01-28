# **Exploring AlexNet for ImageNet Classification**

This project demonstrates the implementation of **AlexNet** for image classification on the ImageNet dataset. By leveraging **transfer learning** and **fine-tuning** techniques, it achieves high accuracy and robust classification performance. The project consolidates key results, metrics, and visualizations into a single Jupyter notebook.

---

## **Features**
- **Model**: AlexNet architecture adapted for 10 ImageNet classes.
- **Performance**: Achieved **93.25% accuracy** with strong precision, recall, and F1-scores.
- **Visualization**: Includes Confusion matrix, Learning curves, Metric breakdowns.
- **Optimization**: Leveraged transfer learning, fine-tuning, and a one-cycle learning policy for efficient training.

---

## **Usage**

1. Open the provided Jupyter notebook: `AlexNet_ImageNet_Classification.ipynb`.
2. Execute the cells sequentially to:
   - Load and preprocess the ImageNet dataset.
   - Train the AlexNet model using transfer learning and fine-tuning.
   - Visualize the results, including performance metrics and training progress.

---

## **Results Summary**

- **Accuracy**: Achieved **93.25%** across 10 ImageNet classes.
- **Metrics**: High precision, recall, and F1-scores across all classes, indicating robust classification performance.
- **Learning Curve**: Consistent training loss reduction and early convergence of validation accuracy.

### **Visualizations**
| Metric      | Value          |
|-------------|----------------|
| Accuracy    | 93.25%         |
| Precision   | High across all classes |
| Recall      | High across all classes |
| F1-Score    | High across all classes |

*(Add confusion matrix and learning curve images here for better clarity.)*

---

## **Insights**

- **Transfer Learning**: Utilized pre-trained AlexNet weights, reducing training time and improving accuracy.
- **Fine-Tuning**: Adaptive learning rates enhanced performance on the target dataset.
- **Efficiency**: Combined transfer learning with the one-cycle learning policy for faster convergence and minimized overfitting.

---

## **Requirements**

To run the notebook, ensure you have the following installed:
- **Python**: Version 3.8 or later
- **Jupyter Notebook**
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
