**Deep Learning Based B-ALL Classification**

***Reference**:*
Ghadezadeh, M., Aria, M., Hosseini, A., & Asadi, F. (2021). A fast and efficient CNN model for B-ALL diagnosis and its subtypes classification using peripheral blood smear images. International Journal of Intelligent Systems. DOI: 10.1002/int.22753.

***Data**: [Acute Lymphoblastic ](https://www.kaggle.com/datasets/mehradaria/leukemia)[Leukemia](https://www.kaggle.com/datasets/mehradaria/leukemia)[ (ALL) image dataset](https://www.kaggle.com/datasets/mehradaria/leukemia)* 

***Task***: Classifying ALL subtypes into-

![](Aspose.Words.e4aa8cc6-e5f1-43ec-87a3-f96d20a864e0.001.png)

***Structure of the Model:***

![](Aspose.Words.e4aa8cc6-e5f1-43ec-87a3-f96d20a864e0.002.png)

- Built the model from scratch in Python using PyTorch
- Pre-processing Block:
  - Decode and Resize: Resized images to 224 x 224 pixels and converted them to Tensors.
  - Normalization: Data was normalized and scaled-down between 0 to 1
  - Augmentation: Tweaking brightness, contrast, rotations, and flips

***Feature Extraction Block: DenseNet-201***

- DenseNet-201 architecture was used as the Feature Extraction block.
- It was initialized with weights pre-trained on the ImageNet dataset.
- The FE takes an image and gives a feature embedding of size 1920.

![](Aspose.Words.e4aa8cc6-e5f1-43ec-87a3-f96d20a864e0.003.png)

***Classification Block:***

- Classifier takes the feature embedding and runs it through two fully connected layers of size 128.

![](Aspose.Words.e4aa8cc6-e5f1-43ec-87a3-f96d20a864e0.004.png)

***Results*:**

- The best model had the following hyperparameters:
  - Learning Rate: 1e-4
  - Number of epochs: 40
  - Optimizer: Adam
  - Regularization: L2 + Dropout

![](Aspose.Words.e4aa8cc6-e5f1-43ec-87a3-f96d20a864e0.005.png)

![](Aspose.Words.e4aa8cc6-e5f1-43ec-87a3-f96d20a864e0.006.png)

- Metrics for the Test Set are as follows:

![](Aspose.Words.e4aa8cc6-e5f1-43ec-87a3-f96d20a864e0.007.png)





