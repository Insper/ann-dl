
!!! success inline end "Deadline and Submission"

    :date: TBD
    
    :clock1: Commits until 23:59

    :material-account-group: [Team (2–3 members)](){ :target="_blank" }

    :simple-github: GitHub Pages link via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

This is an open-themed project where you explore **modern generative models**. You must use at least one architecture from the list below and build a complete generation pipeline. The focus is on understanding the underlying models, not just running demos — you must explain the architecture, the connections between components, and the design choices.

## Eligible Architectures

Choose **at least one** of the following as your primary model:

| Track | Model family | Examples |
|-------|-------------|---------|
| **A** | Latent Diffusion + U-Net | Stable Diffusion 1.5/XL |
| **B** | Flow Matching + DiT | FLUX.1-dev, SD3 |
| **C** | Autoregressive Image Generation | LlamaGen, MaskGIT |
| **D** | Video Generation | CogVideoX, AnimateDiff |
| **E** | Audio Generation | Stable Audio, AudioCraft |
| **F** | Any-to-any Multimodal | Open-source Chameleon variants |

!!! tip "Recommended starting point"
    Track B (FLUX.1) or Track A (Stable Diffusion XL via ComfyUI) are the most accessible while covering the most course content. Track D or E are excellent if your team wants to go further.

## Pipeline Requirements

Your pipeline must chain **at least two model components**. Examples:

- Text → CLIP encoder → FLUX DiT (Flow Matching) → VAE decoder → Image
- Image → Depth estimator → ControlNet + SD → Styled image
- Text → LLM (enhanced prompt) → FLUX → Image → BLIP captioner → refined prompt
- Audio → Whisper → LLM → TTS → new audio
- Text → LLM story → SD image per scene → assembled video

## What You Must Explain

For **each model component** in your pipeline, your report must describe:

1. **Architecture**: what type of network (U-Net, DiT, Transformer, VAR…), number of parameters, key design choices
2. **Training objective**: diffusion loss, flow matching, contrastive, autoregressive, etc.
3. **Role in the pipeline**: what input it receives, what output it produces, why this component is here
4. **Connection to course content**: explicitly link to the relevant lecture (e.g., "This U-Net uses cross-attention as described in the Attention lecture")

!!! danger "No Free Lunch"
    Use only open-source models and free compute (Google Colab, Kaggle, Hugging Face Spaces). Do not use paid APIs (OpenAI, Midjourney, Adobe Firefly). Document GPU hours used.

## Examples of Input–Output Pairs

Provide **at least 8 examples** showing:

- Different text prompts / input styles
- Different inference parameters (CFG scale, number of steps, seed)
- At least 2 failure cases with analysis of why they failed

## Criteria

| Criterion | Description |
|:---------:|-------------|
| **I** | Incomplete delivery or no architecture explanation. |
| **D** | Basic working pipeline with errors; architecture explanation missing or superficial. |
| **C** | One working pipeline (Track A or B) with full architecture explanation for each component. At least 8 input-output examples with varied parameters. |
| **B** | Two working pipelines or one pipeline with advanced techniques (ControlNet, IP-Adapter, LoRA fine-tuning, or video generation). Full architecture documentation. |
| **A** | Grade B plus: custom fine-tuning (LoRA/DreamBooth), original pipeline combining ≥3 components, or a Track D/E implementation. Benchmarked results (FID, CLIP Score, or domain-specific metric). |

A half-grade will be added or subtracted based on report quality, creativity, and depth of architectural analysis.

## Report Structure

Your GitHub Pages report must include:

1. **Introduction**: what pipeline you built and why you chose it
2. **Architecture diagrams**: flow diagrams showing data flow between components (use Mermaid or draw.io)
3. **Component deep-dives**: one section per component with architecture description and math where relevant
4. **Results gallery**: annotated input-output pairs with parameter settings
5. **Failure analysis**: what does not work and why
6. **Reflection**: what you learned, what surprised you, what you would do differently

!!! example "Architecture Diagram Example"
    ```mermaid
    flowchart LR
        A["Text Prompt"] --> B["CLIP Text Encoder\n(ViT-L/14, 123M params)"]
        N["Gaussian Noise\nz~N(0,I)"] --> C
        B --> C["FLUX DiT\n(12B params, Flow Matching)"]
        C -->|"ODE: 20 steps"| D["Clean Latent z₁"]
        D --> E["VAE Decoder\n(83M params)"]
        E --> F["Output Image\n1024×1024px"]
    ```
