
Artificial Intelligence (AI) can be broadly categorized into three main paradigms: Symbolic AI, Connectionist AI, and Neuro-Symbolic AI. Each of these paradigms has its own strengths and weaknesses, and they are often used in different contexts depending on the problem being addressed.

## AI Paradigms
| Paradigm          | Description                                                                                   |
|-------------------|-----------------------------------------------------------------------------------------------|
| Symbolic AI       | Focuses on high-level reasoning and knowledge representation using symbols and rules. It excels in tasks requiring logical reasoning, such as theorem proving and expert systems. However, it struggles with perception and learning from raw data. Examples include logic-based systems, expert systems, and knowledge graphs. |
| Connectionist AI  | Based on artificial neural networks (ANNs), it excels in pattern recognition, learning from large datasets, and handling noisy data. It is particularly effective in tasks like image and speech recognition. However, it often lacks interpretability and struggles with reasoning tasks. Examples include convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers. |
| Neuro-Symbolic AI | Combines the strengths of both symbolic and connectionist AI, aiming to create systems that can reason about complex problems while also learning from data. It leverages symbolic reasoning capabilities alongside neural networks to enhance interpretability and reasoning abilities. Examples include neuro-symbolic systems that integrate symbolic logic with neural networks, such as knowledge-augmented language models and graph neural networks. |

```python exec="on" html="1"
--8<-- "docs/ai/relations.py"
```

Neuro-Symbolic AI combines symbolic reasoning with neural networks, leveraging the strengths of both approaches. It aims to create systems that can reason about complex problems while also learning from data.

This approach is particularly useful in tasks that require both high-level reasoning and the ability to learn from raw data, such as natural language understanding and complex decision-making.

There are several approaches to implementing AI. Machine learning (ML) is one of the most common methods, where algorithms learn from data to make predictions or decisions. Neural networks, a subset of ML, are inspired by the structure and function of the human brain and are particularly effective in tasks like image and speech recognition. Deep learning, a more advanced form of neural networks, uses multiple layers of processing to extract complex patterns from large datasets.

```python exec="on" html="1"
--8<-- "docs/ai/hierarchical.py"
```

## Machine Learning

In the context of AI, machine learning (ML) techniques are used to enable systems to learn from data and improve their performance over time without being explicitly programmed. These techniques allow AI systems to adapt and generalize from examples, making them capable of handling a wide range of tasks, from image recognition to natural language processing.

The techniques are often split into two main categories: **supervised learning** and **unsupervised learning**.

!!! info "Supervised Learning"
    
    **Supervised learning** involves training a model on labeled data, where the input data is paired with the correct output. This allows the model to learn patterns and make predictions based on new, unseen data.
    
    This approach is particularly effective when there is a clear relationship between the input features and the output labels, allowing the model to generalize from the training data to make accurate predictions on new data. Examples include classification tasks (e.g., identifying objects in images) and regression tasks (e.g., predicting house prices based on features).

!!! info "Unsupervised Learning"
    
    **Unsupervised learning**, on the other hand, involves training a model on unlabeled data, where the model must find patterns and relationships within the data without explicit guidance.
    
    This approach is useful for discovering hidden structures in data, such as clusters or groups, without prior knowledge of the labels. It is often used in exploratory data analysis and feature extraction. Examples include clustering tasks (e.g., grouping similar documents) and dimensionality reduction tasks (e.g., reducing the number of features in a dataset while preserving important information).


## Neural Networks

Neural networks are a class of machine learning models inspired by the structure and function of the human brain. They consist of interconnected nodes (neurons) organized in layers, where each connection has an associated weight that is adjusted during training. Neural networks are particularly effective for tasks involving complex patterns, such as image and speech recognition.

Neural networks can be categorized into several types, including: 

- **Feedforward Neural Networks (FNNs)**: The simplest type of neural network where information flows in one direction, from input to output, without cycles. They are commonly used for tasks like classification and regression.
- **Convolutional Neural Networks (CNNs)**: Specialized neural networks designed for processing grid-like data, such as images. They use convolutional layers to automatically learn spatial hierarchies of features, making them highly effective for image recognition tasks.
- **Recurrent Neural Networks (RNNs)**: Neural networks designed for sequential data, such as time series or natural language. They have connections that loop back on themselves, allowing them to maintain a memory of previous inputs. This makes them suitable for tasks like language modeling and speech recognition.
- **Transformers**: A type of neural network architecture that uses self-attention mechanisms to process sequences of data. They have revolutionized natural language processing tasks, enabling models like BERT and GPT to achieve state-of-the-art performance in various language understanding tasks.

## Deep Learning

Deep learning is a subset of machine learning that focuses on using deep neural networks with many layers to learn complex representations of data. It has achieved remarkable success in various domains, including computer vision, natural language processing, and speech recognition. Deep learning models are capable of automatically learning hierarchical features from raw data, eliminating the need for manual feature engineering. This has led to significant advancements in AI applications, enabling systems to perform tasks that were previously considered challenging or impossible.



[^1]: [Wiki - Neuro-Symbolic AI](https://en.wikipedia.org/wiki/Neuro-symbolic_AI){target='_blank'}
[^2]: 2020, Forbes - [Symbolism Versus Connectionism In AI: Is There A Third Way?](https://www.forbes.com/councils/forbestechcouncil/2020/09/01/symbolism-versus-connectionism-in-ai-is-there-a-third-way/){target='_blank'}
[^3]: Garcez, A.d., Lamb, L.C. Neurosymbolic AI: the 3rd wave. Artif Intell Rev 56, 12387–12406 (2023). [doi.org/10.1007/s10462-023-10448-w](https://doi.org/10.1007/s10462-023-10448-w){target='_blank'}

