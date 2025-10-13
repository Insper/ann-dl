!!! success inline end "Deadline and Submission"

    :date: 26.oct (sunday)
    
    :clock1: Commits until 23:59

    :material-account: Individual

    :simple-github: Submission the GitHub Pages' Link (yes, **only** the link for pages) via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

**Activity: VAE Implementation**

In this exercise, you will implement and evaluate a Variational Autoencoder (VAE) on the MNIST or Fashion MNIST dataset. The goal is to understand the architecture, training process, and performance of VAEs.


## Instructions

1. **Data Preparation**:

    - Load the MNIST/Fashion MNIST dataset;
    - Normalize the images to the range [0, 1];
    - Split the dataset into training and validation sets.

2. **Model Implementation**:

    - Define the VAE architecture, including the encoder and decoder networks;
    - Implement the reparameterization trick.

3. **Training**:

    - Train the VAE on the MNIST/Fashion MNIST dataset;
    - Monitor the loss and generate reconstructions during training.

4. **Evaluation**:

    - Evaluate the VAE's performance on the validation set;
    - Generate new samples from the learned latent space.

5. **Visualization**:

    - Visualize original and reconstructed images;
    - Visualize the latent space (in case of a latent space until 3-D, otherwise use a reduced visualization, e.g., using t-SNE, UMAP or PCA).

6. **Report**:

    - Summarize your findings, including challenges faced and insights gained;
    - Include visualizations of reconstructions and latent space.

7. **Extra Credit (Optional)**:

    - Experiment the same dataset with a Autoencoder (AE) and compare the results with the VAE;
    - Experiment with different latent space dimensions and report the effects on reconstruction quality and sample generation.

**Important Notes:**

- The deliverable must be submitted in the format specified: **GitHub Pages**. **No other formats will be accepted.** - there exists a template for the course that you can use to create your GitHub Pages - [template](https://hsandmann.github.io/documentation.template/){target='_blank'};

- There is a **strict policy against plagiarism**. Any form of plagiarism will result in a zero grade for the activity and may lead to further disciplinary actions as per the university's academic integrity policies;

- **The deadline for each activity is not extended**, and it is expected that you complete them within the timeframe provided in the course schedule - **NO EXCEPTIONS** will be made for late submissions.

- **AI Collaboration is allowed**, but each student **MUST UNDERSTAND** and be able to explain all parts of the code and analysis submitted. Any use of AI tools must be properly cited in your report. **ORAL EXAMS** may require you to explain your work in detail.

- All deliverables for individual activities should be submitted through the course platform [insper.blackboard.com](http://insper.blackboard.com/){:target="_blank"}.


**Grade Criteria:**

| Criteria | Description |
|:--------:|-------------|
| **4 pts** | Correctness of the VAE implementation |
| **1 pts** | Training and Evaluation: Proper training procedure, loss monitoring, and evaluation on the validation set. |
| **2 pts** | Sampling: Quality of generated samples. |
| **2 pts** | Visualizations: Quality and clarity of plots (data distribution, decision boundary, accuracy over epochs). |
| **1 pt** | Report Quality: Clarity, organization, and completeness of the report. |
