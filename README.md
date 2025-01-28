# COMP 374 - Deep Learning


Below is a first attempt at a full course on deep learning.
This is still a draft and is subject to revision.


## Course Overview

- **Module 1**: Introduction
    - **Unit 1**: Mathematical Background
- **Module 2**: Neural Networks Basics
    - **Unit 2**: Introduction to Neural Networks
    - **Unit 3**: Training Neural Networks
    - **Unit 4**: Advanced Techniques for Training Neural Networks
    - **Unit 5**: Fundamentals of Convolutional Neural Networks (CNNs)
    - **Unit 6**: Advanced CNN Techniques
- **Module 3**: Midterm
- **Module 4**: Transformers
    - **Unit 7**: Background and Foundations
    - **Unit 8**: Attention and Transformer Architecture
    - **Unit 9**: Applications and Implementation
- **Module 5**: AI Ethics
    - **Unit 10**: Ethical Concerns in AI and Deep Learning
- **Module 6**: Final Exam

# Course Grading Breakdown

The course consists of ten homework assignments, a midterm, and a final.
The point breakdown is

* Homework ($\times$ 10), 70 points each, 700 points total
* Midterm, 150 points
* Final, 150 points

**TOTAL**: 1000 points.

# Links to Course Materials


* Unit 1 Lecture: https://colab.research.google.com/drive/1P4PeoNnpbKffAMEELvzxllDPYoBLIRul
* Unit 1 Homework: https://colab.research.google.com/drive/1v9kLtGyfdQKATi90DZMaIu0_c7urb39x
* Unit 2 Lecture: https://colab.research.google.com/drive/1zySxeu4ykgltnk8oVkeEazoqJ6ET4pZo
* Unit 2 Homework: https://colab.research.google.com/drive/1_J53ma0yX5Ky92BV8VhA3xjs2R992AvA
* Unit 3 Lecture: https://colab.research.google.com/drive/1jB1_uX943FWZ47_rMoIyMUMHZ1MjUo4Z
* Unit 3 Homework: https://colab.research.google.com/drive/1IY6U0-uZW1cwf-BZQMoCAObbNAqgNkgt
* Unit 4 Lecture: https://colab.research.google.com/drive/1IY6U0-uZW1cwf-BZQMoCAObbNAqgNkgt
* Unit 4 Homework: https://colab.research.google.com/drive/1hyZavws_wH39popTib7jpEWV_gdNx87X
* Unit 5 Lecture: https://colab.research.google.com/drive/1j320zGPJMl1imDznSUceVFB7n41lzWU3
* Unit 5 Homework: https://colab.research.google.com/drive/1cRJEOI86OgghAkxoCLGe6YLxDHQCfrAf
* Unit 6 Lecture: https://colab.research.google.com/drive/1u2FkH2Vss8K93XDEyIu-9YwX8JYa8jkS
* Unit 6 Homework: https://colab.research.google.com/drive/1QhV6Z5D1mppu22vp9-OPP-pwG97Hai88
* Unit 7 Lecture: https://colab.research.google.com/drive/1-b0Xy1b8jd0x8T8LQmfoY1Ha1Y3L9HkC
* Unit 7 Homework: https://colab.research.google.com/drive/1bpdT747nFO7MRzpOemqzsmzKBJhFrX-J
* Unit 8 Lecture: https://colab.research.google.com/drive/1bpdT747nFO7MRzpOemqzsmzKBJhFrX-J
* Unit 8 Homework: https://colab.research.google.com/drive/1BritChnHiQuHtopqHS_6HRV2l2MoTNPg
* Unit 9 Lecture: https://colab.research.google.com/drive/1oXQ05dB3z0qsxAvKbeIky3sL5C1vKhcO
* Unit 9 Homework: https://colab.research.google.com/drive/1cvRcUWon3Nh1ObE80sR6b1HzhxQD19gD
* Unit 10 Lecture: https://colab.research.google.com/drive/11n1RRV1qnSgzfuagqFFgZ5V7ZV-mVT6h
* Unit 10 Homework: https://colab.research.google.com/drive/1WePxxyTM8P-zgREJ90UklestfbYmjvlW

---

* Final Exam: https://colab.research.google.com/drive/1GrX9VQZ3sktt7G32XFhYxUtW7m9DWisw
* Midterm Exam: https://colab.research.google.com/drive/1XipW_m8YTwbg5CkB3tmNHLgDuI4T_gG7

# Textbook

* Module 1 covers Chapter 14 and Chapter 2
* Modules 2, 3, and 4 cover Chapter 1
* Modules 5 and 6 cover Chapter 3
* Modules 7, 8 and 9 cover Chapeter 6


# Unit 1: Introduction

## Module 1: Mathematical Background

### Topic Description
An introduction to the mathematical foundations essential for understanding deep learning. This week focuses on derivatives, vector mathematics, matrix multiplication, and basic examples of machine learning models (linear regression and logistic regression). These concepts provide the groundwork for understanding optimization and classification.

### Relevance and Applications
The mathematical concepts introduced this week are essential for the development and understanding of machine learning models:
- **Derivatives** and **gradients** are the backbone of optimization algorithms like gradient descent, which is used in training machine learning models.
- **Vector operations** and **matrix multiplication** are vital for handling multidimensional data and efficient computation in deep learning frameworks.
- **Linear regression** and **logistic regression** serve as stepping stones for understanding more complex models. They are widely used in applications like predicting housing prices, detecting spam emails, and other real-world classification and regression tasks.
- **Jupyter Notebooks** are a current industry standard for data analysis and machine learning model development.

### Module Outcomes
By the end of this week, students should be able to:
1. Compute basic derivatives, including partial derivatives, and understand their significance.
2. Perform vector operations such as dot products and calculate vector norms.
3. Understand and execute matrix multiplication, including its properties and relevance to machine learning models.
4. Implement and understand linear regression and logistic regression as examples of foundational machine learning techniques.
5. Grasp the role of derivatives and gradients in optimization for linear and logistic regression.
6. Understand how to work with Python and Jypyter environments.

### Key Vocabulary
- **Derivative**: A measure of how a function changes as its input changes.
- **Gradient**: A vector representing the direction and rate of fastest increase of a function.
- **Vector**: A quantity with both magnitude and direction, used to represent multidimensional data.
- **Matrix Multiplication**: An operation that combines two matrices to produce a third matrix, essential for linear transformations.
- **Linear Regression**: A model that predicts a continuous output based on input features.
- **Logistic Regression**: A classification model that predicts probabilities of categorical outcomes.
- **Optimization**: The process of finding the best parameters for a model to minimize error.

### Indicators of Mastery
- Solve problems involving derivatives, vector arithmetic, and matrix multiplication.
- Compute gradients for multivariable functions and interpret their geometric significance.
- Implement and explain linear regression and logistic regression models.
- Apply derivatives and matrix operations to optimize simple machine learning models.
- Analyze the performance of linear and logistic regression in basic classification and regression tasks.


# Unit 2: Neural Networks

## Module 2: Introduction to Neural Networks

### Topic Description
This week introduces the concept of neural networks as a general-purpose solution for complex function approximation. Students will explore the types of problems neural networks are designed to solve, the principles behind their design, and the distinction between shallow and deep networks. The session also includes a first hands-on experience with TensorFlow and Keras.

### Relevance and Applications
Neural networks form the backbone of modern deep learning. They are:
- **Powerful function approximators** capable of fitting highly complex datasets.
- **Widely applicable**, addressing challenges in image recognition, natural language processing, recommendation systems, and more.
- **Scalable**, enabling the creation of shallow or deep models to meet specific performance and complexity needs. 

Learning the basics of neural networks sets the stage for exploring more advanced architectures and applications in subsequent weeks.

### Module Outcomes
By the end of this week, students should be able to:
1. Understand the types of problems neural networks aim to solve and the requirements of a general-purpose function approximator.
2. Explain why neural networks are effective for solving complex, nonlinear problems.
3. Conceptualize neural networks as mathematical functions and understand their structure.
4. Differentiate between shallow and deep networks and their respective use cases.
5. Familiarize themselves with TensorFlow and Keras as tools for building and training neural networks.

### Key Vocabulary
- **Neural Network**: A computational model inspired by the structure of the brain, used to approximate complex functions.
- **Activation Function**: A mathematical function applied to a neuron's output, introducing nonlinearity to the model.
- **Shallow Network**: A neural network with fewer layers, typically easier to train but less capable of modeling complex relationships.
- **Deep Network**: A neural network with many layers, capable of modeling intricate and hierarchical data patterns.
- **General-Purpose Function Approximator**: A model capable of fitting a wide variety of data distributions and patterns.
- **TensorFlow**: An open-source machine learning framework for building and training models.
- **Keras**: A high-level API in TensorFlow designed for easy and rapid model building.
- **Universal Approximation Theorem**: A mathematical theorem stating that a feed-forward neural network with at least one hidden layer and a non-linear activation function can approximate any continuous function to arbitrary precision, given sufficient neurons in the hidden layer.

### Indicators of Mastery
- Describe neural networks as mathematical functions and explain their advantages for solving nonlinear problems.
- Differentiate between shallow and deep networks, articulating their strengths and limitations.
- Build and train a basic neural network using TensorFlow and Keras.
- Explain the role of activation functions and their importance in introducing nonlinearity.
- Evaluate the performance of a neural network and identify scenarios where neural networks are an appropriate solution.


### Supplemental Material

[But what is a neural network?](https://youtu.be/aircAruvnKk?si=6PUfl3KnBts4Pg5Z)

## Module 3: Introduction to Training Neural Networks

### Topic Description
This week introduces the process of training neural networks, focusing on foundational optimization techniques such as gradient descent and stochastic gradient descent. Students will also learn about train/test splits and learning curves to evaluate model performance. Hands-on practice with training basic neural networks in Keras will reinforce these concepts.

### Relevance and Applications
Training is a critical step in the development of neural networks:
- **Optimization techniques** like gradient descent are essential for enabling neural networks to learn from data.
- **Evaluation strategies** such as train/test splits and learning curves help understand model performance and ensure reliable predictions.

Mastering these foundational concepts prepares students to fine-tune and improve models effectively.

### Module Outcomes
By the end of this week, students should be able to:
1. Explain the concept of training in the context of neural networks.
2. Understand and apply gradient descent and stochastic gradient descent to optimize neural networks.
3. Use train/test splits and interpret learning curves to evaluate model performance.

### Key Vocabulary
- **Gradient Descent**: An optimization algorithm that iteratively adjusts model parameters to minimize the loss function.
- **Stochastic Gradient Descent (SGD)**: A variation of gradient descent that updates parameters using a single or small batch of data points.
- **Train/Test Split**: A method of dividing data into separate training and testing sets to evaluate model performance.
- **Learning Curve**: A graph showing the change in a model's performance metric over time during training.
- **Loss Function**: A function with the property that the smaller its output, the better performing the model is.

### Indicators of Mastery
- Describe the training process for neural networks and its importance.
- Implement gradient descent and stochastic gradient descent for optimization.
- Evaluate model performance using train/test splits and learning curves.

### Supplemental Material

[Gradient descent, how neural networks learn](https://youtu.be/IHZwWFHWa-w?si=wLhXyjxnJxcic3Ao)

## Module 4: Advanced Techniques for Training Neural Networks

### Topic Description
Building on the basics of training, this week introduces advanced techniques to optimize neural network performance. 
Topics include hyperparameter tuning, dropout, optimizers, and L1/L2 regularization to prevent overfitting.
 Students will also learn how to configure and fine-tune neural networks using Keras.

### Relevance and Applications
Advanced training techniques ensure neural networks generalize well to new data:
- **Hyperparameter tuning** improves model performance and efficiency.
- **Regularization techniques** like dropout and L1/L2 help mitigate overfitting, ensuring models perform well on unseen data.

These advanced methods are vital for creating robust and reliable machine learning models.

### Module Outcomes
By the end of this week, students should be able to:
1. Identify and adjust key hyperparameters, including the number of layers, size of layers, and learning rate.
2. Understand and apply regularization techniques like dropout and L1/L2 regularization to prevent overfitting.
3. Train and evaluate advanced neural networks using Keras.

### Key Vocabulary
- **Hyperparameter**: Configurable parameters that are set before training begins, such as the number of layers or the learning rate.
- **Learning Rate**: A hyperparameter that controls the size of steps taken during gradient descent.
- **Dropout**: A regularization technique that randomly ignores a subset of neurons during training to prevent overfitting.
- **L1 Regularization**: A technique that adds a penalty proportional to the absolute value of the weights to the loss function, encouraging sparsity.
- **L2 Regularization**: A technique that adds a penalty proportional to the square of the weights to the loss function, encouraging smaller weights.

### Indicators of Mastery
- Adjust hyperparameters, such as the number and size of layers, to improve model performance.
- Apply dropout and L1/L2 regularization to reduce overfitting.
- Train and evaluate advanced neural networks using Keras.


# Module 5: Fundamentals of Convolutional Neural Networks (CNNs)

## Topic Description
This week introduces the foundational concepts of convolutional neural networks (CNNs). Students will learn the principles of convolution and the role of convolutional and pooling layers. The focus is on building and training CNNs for basic image classification tasks using Keras.

## Relevance and Applications
CNNs are a key architecture for handling image data, with applications such as:
- **Image classification**: Assigning labels to images based on their content.
- **Medical imaging**: Identifying patterns and abnormalities in radiology scans.

Mastering the basics of CNNs prepares students to tackle more advanced tasks like object detection and multi-input models.

## Module Outcomes
By the end of this week, students should be able to:
1. Understand convolution and its application in feature extraction.
2. Explain the structure and purpose of convolutional and pooling layers.
3. Build and train a basic CNN for image classification using Keras.

## Key Vocabulary
- **Convolution**: A mathematical operation used in CNNs to extract features from input data by applying filters.
- **Convolutional Layer**: A layer in a CNN that performs convolution operations to detect features like edges and textures.
- **Pooling Layer**: A layer that reduces the spatial dimensions of feature maps to reduce computation and extract dominant features.
- **ReLU (Rectified Linear Unit)**: An activation function that introduces nonlinearity by outputting the positive part of input values.

## Indicators of Mastery
- Explain the purpose and structure of convolutional and pooling layers.
- Implement a CNN for basic image classification using Keras.
- Compare CNNs with feed-forward neural networks for image-based tasks.

### Supplemental Material

[But what is colvolution?](https://youtu.be/KuXjwB4LzSA?si=7rm6seM0qYomHXhD)

# Module 6: Advanced CNN Techniques

## Topic Description
Building on the basics, this week covers advanced techniques for CNNs, including weight visualization for understanding learned features and object detection using bounding rectangles. Students will also learn to implement multi-input models, such as combining images with text or other features, using concatenation layers.

## Relevance and Applications
Advanced CNN techniques enable more complex real-world applications:
- **Weight visualization** provides insights into how CNNs learn to detect features like edges and shapes.
- **Object detection** is critical in fields such as video analysis and autonomous vehicles.
- **Multi-input models** are used in applications requiring combined data sources, such as combining images with metadata.

These techniques expand the versatility and interpretability of CNN-based models.

## Module Outcomes
By the end of this week, students should be able to:
1. Visualize CNN weights to interpret learned features.
2. Implement object detection tasks with bounding rectangles.
3. Build and train multi-input models using concatenation layers in Keras.

## Key Vocabulary
- **Weight Visualization**: A technique for inspecting the filters learned by CNNs to understand how they process input data.
- **Bounding Rectangle**: A box drawn around an object in an image to indicate its detected location.
- **Object Detection**: The task of identifying and localizing objects within an image.
- **Concatenation Layer**: A neural network layer which combines tensors into one larger tensor for multi-input models.

## Indicators of Mastery
- Visualize and interpret CNN weights to understand feature learning.
- Implement object detection with bounding rectangles for identifying objects in images.
- Build and train multi-input models using concatenation layers in Keras.


# Midterm

## Exam Overview
The midterm exam assesses knowledge and skills from the first four modules of the course, covering foundational mathematical concepts, the basics of neural networks, training methodologies, and convolutional neural networks (CNNs). The exam is designed to evaluate theoretical understanding and practical application through a combination of multiple-choice, short-answer, and Python programming tasks.

### Exam Sections

#### **Section 1: Mathematical Background**
- **Focus**: Assessing foundational mathematical skills necessary for deep learning.
- **Question Types**:
  - Solve derivatives and compute gradients for given functions.
  - Perform matrix operations and explain their relevance in neural networks.

#### **Section 2: Neural Networks**
- **Focus**: Testing knowledge of training techniques and model optimization.
- **Question Types**:
  - Explain the training process and its challenges.
  - Implement a neural network with dropout and evaluate its performance.

#### **Section 3: Convolutional Neural Networks**
- **Focus**: Assessing understanding of CNNs and their application to image-based tasks.
- **Question Types**:
  - Explain the role of convolutional layers in feature extraction.
  - Implement a simple object detection task in Python.


### Exam Format
1. **Multiple-Choice Questions (20%)**: Test theoretical knowledge across all topics.
2. **Short-Answer Questions (30%)**: Evaluate conceptual understanding and problem-solving skills.
3. **Programming Tasks (50%)**: Hands-on tasks requiring Python implementation of neural networks and training techniques.

# Unit 3: Transformers

## Module 7: Background and Foundations

### Topic Description
This week provides the foundational knowledge necessary for understanding transformers. Students will explore sequential data processing challenges with RNNs, the limitations these architectures face, and the revolutionary introduction of attention mechanisms. The session includes cosine similarity for measuring vector relationships and embedding layers for representing categorical data.

### Module Outcomes
By the end of this week, students should be able to:
1. Explain the limitations of RNNs in handling long sequences and parallel processing.
2. Understand the concept of attention and its role in overcoming RNN challenges.
3. Calculate cosine similarity and apply it to understand relationships between vectors.
4. Implement embedding layers for transforming categorical data into continuous representations.

### Key Vocabulary
- **RNN (Recurrent Neural Network)**: An architecture designed for sequential data but prone to issues like vanishing gradients.
- **Attention Mechanism**: A method that allows models to focus on relevant parts of input sequences.
- **Cosine Similarity**: A metric for measuring the cosine of the angle between two vectors, indicating their similarity.
- **Embedding Layer**: A neural network layer that maps categorical data to continuous, dense vector representations.

### Indicators of Mastery
- Explain why RNNs face limitations with long sequences.
- Demonstrate the calculation and significance of cosine similarity in vector spaces.
- Build an embedding layer in Keras for categorical data.
- Articulate the motivation and basic mechanics of attention.

## Module 8: Attention and Transformer Architecture

### Topic Description
This week dives deeper into the attention mechanism and introduces the core architecture of transformers. Students will learn about self-attention, multi-head attention, and positional encoding. Batch normalization is introduced as a key technique for stabilizing training.

### Module Outcomes
By the end of this week, students should be able to:
1. Explain self-attention and multi-head attention in detail.
2. Understand the architecture of a transformer and its components.
3. Apply positional encoding to preserve sequence information.
4. Implement batch normalization to improve model stability.

### Key Vocabulary
- **Self-Attention**: A mechanism enabling models to weigh the importance of each element in a sequence relative to the others.
- **Query, Key, Value**: Components of the attention mechanism where queries identify what to focus on, keys provide context, and values hold the data being processed.
- **Multi-Head Attention**: An extension of self-attention that uses multiple parallel attention mechanisms for richer representation.
- **Positional Encoding**: A technique to introduce order into input sequences for transformers.
- **Batch Normalization**: A method to standardize inputs to a layer, improving convergence and stability.

### Indicators of Mastery
- Calculate and interpret self-attention weights using query, key, and value.
- Implement multi-head attention in Keras.
- Demonstrate the use of positional encoding in a transformer model.
- Apply batch normalization to stabilize and improve transformer training.



## Module 9: Applications and Implementation

### Topic Description
In the final week, students will implement transformers in Keras and explore their applications in natural language processing (e.g., text classification, machine translation) and computer vision (e.g., vision transformers). The week includes fine-tuning pre-trained transformers for real-world tasks.

### Module Outcomes
By the end of this week, students should be able to:
1. Implement a transformer model from scratch using Keras.
2. Fine-tune pre-trained transformers for specific tasks.
3. Apply transformers to solve real-world problems in NLP and computer vision.
4. Evaluate transformer models using appropriate performance metrics.

### Key Vocabulary
- **Pre-trained Transformers**: Models like BERT and GPT, trained on large datasets and adaptable to specific tasks through fine-tuning.
- **Vision Transformers (ViT)**: Transformer models adapted for image classification and other vision tasks.
- **Fine-Tuning**: Customizing a pre-trained model to a specific task by continuing training on new data.

### Indicators of Mastery
- Build and train a transformer model in Keras.
- Fine-tune pre-trained transformers for NLP and vision applications.
- Evaluate the performance of transformers on real-world datasets.
- Apply transformers to diverse tasks, demonstrating their versatility and power.

# Unit 4: AI Ethics

## Module 10: Ethical Concerns in AI and Deep Learning

### Topic Description
This week addresses the ethical implications of deploying AI systems, focusing on the limitations of AI models, institutional bias, bias in training data, and AI alignment. Students will explore real-world case studies and discuss strategies for developing responsible AI systems that align with societal values and ethical principles.

### Relevance and Applications
Understanding ethical concerns is critical for the responsible development and deployment of AI systems:
- **AI Model Limitations**: Recognizing where and why AI fails helps prevent over-reliance and misuse.
- **Institutional Bias and Data Bias**: Identifying and mitigating bias ensures fairness in AI decision-making processes, such as hiring or lending.
- **AI Alignment**: Ensuring AI systems operate in line with human intentions and societal values is essential for long-term safety and trust.

By exploring these issues, students will gain the skills to analyze and address ethical challenges in their AI projects.

### Module Outcomes
By the end of this week, students should be able to:
1. Identify limitations in AI models and understand their implications for decision-making.
2. Analyze the sources and impact of institutional and data biases in AI systems.
3. Discuss AI alignment and its significance in developing systems that align with human values.
4. Propose strategies for improving fairness, transparency, and accountability in AI systems.
5. Evaluate real-world AI systems through an ethical lens.

### Key Vocabulary
- **AI Model Limitations**: Constraints in AI systems, such as generalization errors, lack of interpretability, and failure in edge cases.
- **Institutional Bias**: Systemic bias arising from policies or practices embedded in organizations and reflected in AI outputs.
- **Data Bias**: Bias in training datasets that leads to discriminatory or unfair AI predictions.
- **AI Alignment**: The process of ensuring AI systems' goals and behaviors align with human values and intentions.
- **Transparency**: The degree to which AI models and their decision-making processes can be understood by humans.
- **Fairness**: The quality of making decisions that are free of discrimination or favoritism.

### Indicators of Mastery
- Identify and explain limitations of AI models in real-world scenarios.
- Analyze the presence and impact of bias in AI datasets and institutional practices.
- Evaluate AI systems for alignment with human values and societal goals.
- Propose actionable solutions to improve fairness, transparency, and accountability in AI projects.
- Critically assess the ethical implications of deploying AI systems in various domains.

This module provides students with the tools to critically evaluate and address the ethical challenges posed by AI, preparing them for responsible AI development in their future careers.

### Supplamental Material

https://www.youtube.com/watch?v=bJLcIBixGj8&t=63s
https://www.youtube.com/watch?v=qV_rOlHjvvs

# Final Exam

## Exam Overview
The final exam evaluates students' understanding of key topics covered throughout the semester, with a primary focus on transformers. It includes a review of foundational concepts from the first half of the course, advanced transformer architectures and applications, and ethical considerations in AI development. The exam comprises theoretical questions, programming tasks, and case-based analysis to ensure a comprehensive assessment of both knowledge and practical skills.

## Exam Sections

### **Section 1: Foundational Concepts (15%)**
- **Focus**: Revisiting foundational topics from the first half of the semester.
- **Key Topics**:
  - Gradient descent and training neural networks.
  - Basics of convolutional neural networks (CNNs).
  - Understanding embedding layers for categorical data.
- **Question Types**:
  - Explain the role of gradients in optimization.
  - Implement a CNN for a basic image classification task.
  - Use an embedding layer to preprocess categorical data.

### **Section 2: Transformers (65%)**
- **Focus**: Comprehensive evaluation of transformer architectures and applications.
- **Key Topics**:
  - Self-attention and multi-head attention mechanisms.
  - Transformer architecture and positional encoding.
  - Implementation of transformers in Keras.
  - Fine-tuning pre-trained transformers for specific tasks.
  - Applications of transformers in NLP and computer vision.
- **Question Types**:
  - Calculate self-attention weights using query, key, and value vectors.
  - Explain the purpose of multi-head attention and positional encoding.
  - Implement a transformer model in Keras for text classification.
  - Fine-tune a pre-trained transformer (e.g., BERT) for sentiment analysis.
  - Discuss the advantages of transformers over RNNs for sequence data.

### **Section 3: Ethical Considerations (20%)**
- **Focus**: Applying ethical principles to AI and deep learning projects.
- **Key Topics**:
  - AI model limitations.
  - Institutional and data bias.
  - AI alignment with human values.
- **Question Types**:
  - Analyze a real-world AI system for bias in its data and outputs.
  - Identify ethical challenges in deploying transformers for sensitive tasks (e.g., hiring, medical diagnosis).
  - Propose strategies for improving fairness and transparency in an AI application.

## Exam Format
1. **Multiple-Choice Questions (20%)**: Test theoretical understanding across all topics.
2. **Short-Answer Questions (30%)**: Assess the ability to explain concepts and solve theoretical problems.
3. **Programming Tasks (40%)**: Evaluate hands-on skills, focusing on transformers and key first-half concepts.
4. **Case Study Analysis (10%)**: Analyze a real-world scenario involving ethical considerations in AI.

## Exam Preparation Tips
- Review key concepts from Weeks 1–4, including gradient descent, CNNs, and embedding layers.
- Practice implementing transformers and their components (self-attention, multi-head attention) using Keras.
- Work on fine-tuning pre-trained transformers for diverse NLP and vision tasks.
- Reflect on ethical case studies and develop strategies for addressing common challenges like bias and transparency.


This final exam ensures a well-rounded assessment of the students' technical skills, conceptual understanding, and ethical awareness.

