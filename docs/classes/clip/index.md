!!! tip inline end "More about NLP"

    Check out Tiago Tavares' NLP course that covers Transformers and other advanced NLP topics: [https://tiagoft.github.io/nlp_course/](https://tiagoft.github.io/nlp_course/){:target="_blank"}.

CLIP (Contrastive Language-Image Pretraining) is a multimodal machine learning model developed by OpenAI in 2021. It bridges the gap between vision and language by jointly training an image encoder and a text encoder on a massive dataset of image-text pairs scraped from the internet (around 400 million pairs). The core idea is to learn representations where images and their corresponding textual descriptions are embedded close together in a shared latent space, while non-matching pairs are pushed apart. This enables zero-shot learning capabilities, meaning CLIP can perform tasks like image classification without being explicitly trained on labeled data for those tasks—simply by comparing image embeddings to text embeddings of class descriptions.

## Key Components:

- **Image Encoder**: Typically a Vision Transformer (ViT) or a modified ResNet that processes images into fixed-dimensional embeddings (e.g., 512 or 768 dimensions).
- **Text Encoder**: A Transformer-based model (like a modified GPT or BERT variant) that encodes text captions into embeddings of the same dimensionality.
- **Training Objective**: Contrastive loss (specifically, a symmetric version of InfoNCE loss). For a batch of N image-text pairs, it computes a similarity matrix between all image and text embeddings, treats the diagonal (matching pairs) as positives, and off-diagonals as negatives. The goal is to maximize similarity for positives and minimize for negatives.
- **Inference**: To classify an image, encode it and compare its embedding (via cosine similarity) to encoded text prompts like =="a photo of a [class]"==. The highest similarity wins.

![](overview-a.svg){width="70%"}
/// caption
CLIP architecture overview. During training, image and text encoders are trained jointly with contrastive loss on image-text pairs. (from OpenAI's CLIP paper[^2])
///

![](overview-b.svg){width="70%"}
/// caption
CLIP architecture overview. At inference, image embeddings are compared to text embeddings of class prompts for zero-shot classification. (from OpenAI's CLIP paper[^2])
///

CLIP's strength lies in its scalability and generalization. It doesn't require task-specific fine-tuning and can handle open-vocabulary tasks, but it has limitations like sensitivity to prompt engineering and biases from internet data.

> An ImageNet model is good at predicting the 1000 ImageNet categories, but that’s all it can do “out of the box.” If we wish to perform any other task, an ML practitioner needs to build a new dataset, add an output head, and fine-tune the model. In contrast, CLIP can be adapted to perform a wide variety of visual classification tasks without needing additional training examples. To apply CLIP to a new task, all we need to do is “tell” CLIP’s text-encoder the names of the task’s visual concepts, and it will output a linear classifier of CLIP’s visual representations. The accuracy of this classifier is often competitive with fully supervised models.[^1]

!!! quote "[Limitations of CLIP](https://openai.com/research/clip#limitations)"

    While CLIP usually performs well on recognizing common objects, it struggles on more abstract or systematic tasks such as counting the number of objects in an image and on more complex tasks such as predicting how close the nearest car is in a photo. On these two datasets, zero-shot CLIP is only slightly better than random guessing. Zero-shot CLIP also struggles compared to task specific models on very fine-grained classification, such as telling the difference between car models, variants of aircraft, or flower species.

    CLIP also still has poor generalization to images not covered in its pre-training dataset. For instance, although CLIP learns a capable OCR system, when evaluated on handwritten digits from the MNIST dataset, zero-shot CLIP only achieves 88% accuracy, well below the 99.75% of humans on the dataset. Finally, we’ve observed that CLIP’s zero-shot classifiers can be sensitive to wording or phrasing and sometimes require trial and error “prompt engineering” to perform well.[^1]



## Numerical Simulation of CLIP's Contrastive Loss

To illustrate how CLIP works numerically, let's simulate a tiny batch with 3 image-text pairs. We'll assume pre-computed embeddings (in practice, these come from the encoders). Each embedding is a 3D vector for simplicity (real CLIP uses higher dimensions like 512).

#### Setup:

- Image embeddings (I):  

    \( I_1 = [1.0, 0.0, 0.0] \)  (e.g., for "cat")  
    \( I_2 = [0.0, 1.0, 0.0] \)  (e.g., for "dog")  
    \( I_3 = [0.0, 0.0, 1.0] \)  (e.g., for "bird")

- Text embeddings (T):  

    \( T_1 = [0.9, 0.1, 0.0] \)  (close to $I_1$)  
    \( T_2 = [0.1, 0.8, 0.1] \)  (close to $I_2$)  
    \( T_3 = [0.0, 0.3, 0.7] \)  (close to $I_3$)  

- Batch size ($N$): $3$

- Temperature ($\tau$): $0.07$ (a hyperparameter to scale logits; common in CLIP).

#### Step-by-Step Calculation:

1. **Normalize Embeddings**:

    CLIP uses L2-normalized embeddings for cosine similarity. Here, they're already unit-length for simplicity (assume they are).

2. **Compute Similarity Matrix (Logits)**:

    Similarity = \( \displaystyle \frac{(I \cdot T)}{\tau} \)  (dot product scaled by τ).

    Calculations:

    \(
    \begin{align*}
    \text{Logits}_{I \to T} &= \begin{bmatrix}
    \text{sim}(I_1, T_1) & \text{sim}(I_1, T_2) & \text{sim}(I_1, T_3) \\
    \text{sim}(I_2, T_1) & \text{sim}(I_2, T_2) & \text{sim}(I_2, T_3) \\
    \text{sim}(I_3, T_1) & \text{sim}(I_3, T_2) & \text{sim}(I_3, T_3)
    \end{bmatrix} \\
    &= \begin{bmatrix}
    \frac{1 \cdot 0.9 + 0 \cdot 0.1 + 0 \cdot 0.0}{0.07} & \frac{1 \cdot 0.1 + 0 \cdot 0.8 + 0 \cdot 0.1}{0.07} & \frac{1 \cdot 0.0 + 0 \cdot 0.3 + 0 \cdot 0.7}{0.07} \\
    \frac{0 \cdot 0.9 + 1 \cdot 0.1 + 0 \cdot 0.0}{0.07} & \frac{0 \cdot 0.1 + 1 \cdot 0.8 + 0 \cdot 0.1}{0.07} & \frac{0 \cdot 0.0 + 1 \cdot 0.3 + 0 \cdot 0.7}{0.07} \\
    \frac{0 \cdot 0.9 + 0 \cdot 0.1 + 1 \cdot 0.0}{0.07} & \frac{0 \cdot 0.1 + 0 \cdot 0.8 + 1 \cdot 0.1}{0.07} & \frac{0 \cdot 0.0 + 0 \cdot 0.3 + 1 \cdot 0.7}{0.07}
    \end{bmatrix} \\
    &\approx \begin{bmatrix}
    12.857 &  1.4286 &  0 \\
    1.4286 & 11.4286 &  4.2857 \\
    0 &  1.4286 & 10
    \end{bmatrix}
    \end{align*}
    \)

    Full image-to-text logit matrix:  

    \(
    \text{Logits}_{I \to T} \approx \begin{bmatrix}
    12.857 &  1.4286 &  0 \\
    1.4286 & 11.4286 &  4.2857 \\
    0 &  1.4286 & 10
    \end{bmatrix}
    \)

    CLIP averages both directions, text-to-image logits are the transpose:
    
    \[
    \text{Logits}_{T \to I} = \text{Logits}_{I \to T}^T
    \]

3. **Softmax for Probabilities**:

    For each row (image), softmax over logits to get probabilities of matching texts.  

    \(
    \displaystyle \text{Softmax}(I) = \frac{e^{I_i}}{\sum_{j} e^{I_j}}
    \)

    Calculating exponentials and normalizing:

    \(
    \begin{align*}
    \sum_{j} e^{I_j} &\approx \begin{bmatrix}
    e^{12.857} +  e^{1.4286} + e^{0} \\
    e^{1.4286} +  e^{11.4286} + e^{4.2857} \\
    e^{0} +  e^{1.4286} + e^{10}
    \end{bmatrix} \\
    &\approx \begin{bmatrix}
    383523 \\
    91987 \\
    22031
    \end{bmatrix}
    \end{align*}
    \)

    Then:
    
    \(
    \begin{align*}
    \text{Softmax}(I) &\approx \begin{bmatrix}
    \frac{e^{12.857}}{383523} &  \frac{e^{1.4286}}{383523} &  \frac{e^{0}}{383523} \\
    \frac{e^{1.4286}}{91987} &  \frac{e^{11.4286}}{91987} &  \frac{e^{4.2857}}{91987} \\
    \frac{e^{0}}{22031} &  \frac{e^{1.4286}}{22031} &  \frac{e^{10}}{22031}
    \end{bmatrix} \\
    &\approx \begin{bmatrix}
    0.9999 & 0 & 0 \\
    0 & 0.9992 & 0.0008 \\
    0 & 0.0002 & 0.9998
    \end{bmatrix}
    \end{align*}
    \)

    The diagonal should have high probs.

4. **Contrastive Loss**:

    Negative log-likelihood of correct labels (diagonal).  

    \[
    \mathcal{L}_{I \to T} = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{i \to t})
    \]

    For this batch:  

    \( \mathcal{L}_{I_1 \to T_1} = \log(0.9999) \approx -0.0000 \)

    \( \mathcal{L}_{I_2 \to T_2} = \log(0.9992) \approx -0.0004 \)

    \( \mathcal{L}_{I_3 \to T_3} = \log(0.9998) \approx -0.0001 \)

    $\mathcal{L}_{I \to T} \approx 0.00016$ (very low loss since embeddings are well-aligned).

    CLIP computes symmetric loss:

    \[
    \displaystyle \mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{I \to T} + \mathcal{L}_{T \to I} \right).
    \]

    In training, gradients update encoders to minimize this. If embeddings were misaligned (e.g., I1 close to T2), loss would be higher.

{==

This is a simplified simulation; real CLIP handles large batches (e.g., 32k) and uses distributed training.

==}

---

## Additional


### L2-normalized embeddings

L2-normalized embeddings are vectors whose length is scaled to a unit of 1, meaning their L2 norm (Euclidean length) is equal to one. This is achieved by dividing each component of the original vector by its total L2 norm, making it a common method for ensuring consistent magnitude and improving the effectiveness of distance-based similarity measures like cosine similarity[^4]. 

**How it works**

1. **Calculate the L2 norm**:

    For a vector \(v=[v_{1},v_{2},...,v_{n}]\), the L2 norm (\(||v||_{2}\)) is the square root of the sum of the squares of its components: \(||v||_{2}=\sqrt{v_{1}^{2}+v_{2}^{2}+...+v_{n}^{2}}\).
    
2. **Divide each component**:

    Each element of the vector is then divided by this calculated L2 norm. The resulting normalized vector, \(v^{\prime }\), is:
    
    \(v^{\prime }=[\frac{v_{1}}{||v||_{2}},\frac{v_{2}}{||v||_{2}},...,\frac{v_{n}}{||v||_{2}}]\). 

**Why it is used**

- **Focus on direction**: It helps models focus on the "direction" of the vector in a high-dimensional space rather than its magnitude, which can be useful when the magnitude doesn't carry meaningful information. 
- **Improves similarity measures**: Normalization is crucial for techniques that rely on cosine similarity. L2-normalized embeddings make the similarity score equal to the dot product, simplifying calculations and comparison. 
- **Prevents magnitude bias**: It ensures that embeddings with large magnitudes don't dominate similarity comparisons, preventing bias from large values. 
- **Used in model architecture**: Some models use L2 normalization as a constraint to keep embeddings on a hypersphere, which can be beneficial for tasks like face recognition or out-of-distribution detection. 

**When to use it**

- When using cosine similarity for tasks like retrieval or recommendation. 
- In deep learning models where the magnitude of the weights can grow uncontrollably and affect performance. 
- When you want to constrain the representation space to a sphere, as it can lead to more stable training. 






[^1]: [CLIP: Connecting Text and Images](https://openai.com/index/clip){:target="_blank"}

[^2]: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020){:target="_blank"}, Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever, 2021.

[^3]: [How to Normalize a Vector](https://nextbridge.com/learn-how-to-normalize-a-vector/){:target="_blank"}, Nextbridge.

[^4]: [Cosine Similarity](https://www.geeksforgeeks.org/dbms/cosine-similarity/){:target="_blank"}, GeeksforGeeks.
