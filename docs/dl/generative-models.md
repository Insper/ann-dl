Generative models are a class of machine learning models that learn to generate new data samples that resemble a given training dataset. They are particularly useful in tasks such as image synthesis, text generation, and audio generation. Generative models can be broadly categorized into two main types: explicit generative models and implicit generative models.

- **Explicit generative models** define a probability distribution over the data and can sample from this distribution to generate new data points. Examples include Gaussian Mixture Models (GMMs) and Variational Autoencoders (VAEs). 

- **Implicit generative models**, on the other hand, do not explicitly define a probability distribution but instead learn to generate samples directly through mechanisms like adversarial training. Generative Adversarial Networks (GANs) are a prominent example of implicit generative models.

Additionally, there are **diffusion models**, which have gained popularity in recent years for their ability to generate high-quality images and other data types. These models work by gradually transforming a simple noise distribution into a complex data distribution through a series of denoising steps, allowing them to capture intricate details and variations in the data.


## Additional Resources

- [GAN Lab](https://poloclub.github.io/ganlab/){target='_blank'} is an interactive visualization tool that helps users understand how Generative Adversarial Networks (GANs) work. It provides a hands-on experience of training GANs, allowing users to visualize the generator and discriminator networks, observe their interactions, and see how they evolve during the training process. This tool is particularly useful for those new to GANs or for educators looking to demonstrate the concepts behind adversarial training in a more intuitive way.
