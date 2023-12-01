# Deep Learning Based B-ALL Classification

***Reference**:*
Ghadezadeh, M., Aria, M., Hosseini, A., & Asadi, F. (2021). A fast and efficient CNN model for B-ALL diagnosis and its subtypes classification using peripheral blood smear images. International Journal of Intelligent Systems. DOI: 10.1002/int.22753.

***Data**: [Acute Lymphoblastic ](https://www.kaggle.com/datasets/mehradaria/leukemia)[Leukemia](https://www.kaggle.com/datasets/mehradaria/leukemia)[ (ALL) image dataset](https://www.kaggle.com/datasets/mehradaria/leukemia)* 

***Task***: Classifying ALL subtypes into-

![image](https://github.com/sanyuktaadap/leukemia-classification/assets/126644146/662adbf1-84b0-4218-b662-df74a4f30cd7)



***Structure of the Model:***

![image](https://github.com/sanyuktaadap/leukemia-classification/assets/126644146/0e97dd0b-c46a-4c3e-8613-d3dc87d26b2e)



- Built the model from scratch in Python using PyTorch
- Pre-processing Block:
  - Decode and Resize: Resized images to 224 x 224 pixels and converted them to Tensors.
  - Normalization: Data was normalized and scaled-down between 0 to 1
  - Augmentation: Tweaking brightness, contrast, rotations, and flips

***Feature Extraction Block: DenseNet-201***

- DenseNet-201 architecture was used as the Feature Extraction block.
- It was initialized with weights pre-trained on the ImageNet dataset.
- The FE takes an image and gives a feature embedding of size 1920.

![image](https://github.com/sanyuktaadap/leukemia-classification/assets/126644146/6ed4a376-f36c-4746-97ef-a036e53da127)



***Classification Block:***

- Classifier takes the feature embedding and runs it through two fully connected layers of size 64.

![image](https://github.com/sanyuktaadap/leukemia-classification/assets/126644146/06948179-1df5-4ee9-8e97-ae0d462d7b02)



***Results*:**

- The best model had the following hyperparameters:
  - Learning Rate: 1e-4
  - Number of epochs: 40
  - Optimizer: Adam
  - Regularization: L2 + Dropout

Training Loss
![image](https://github.com/sanyuktaadap/leukemia-classification/assets/126644146/fa4caba1-2d91-46cc-912d-72560a75412d)

Validation Loss
![image](https://github.com/sanyuktaadap/leukemia-classification/assets/126644146/3c07efc0-bddf-4d6c-9498-ccec8538749c)



- Metrics for the Test Set are as follows:
  - Accuracy: 99.6%
  - Specificity: 99.9%
  - Precision: 98.0%
  - Recall: 97.8%
  - F1 Score: 97.8%







