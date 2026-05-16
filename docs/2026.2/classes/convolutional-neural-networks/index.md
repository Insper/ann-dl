### Introduction to Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a class of deep neural networks commonly used for image recognition, video analysis, and other tasks involving grid-like data (e.g., pixels in images). Unlike fully connected networks, CNNs exploit spatial hierarchies through **convolutions**, which apply learnable filters (kernels) to local regions of the input. This reduces parameters, enables translation invariance, and captures features like edges, textures, or objects.

A typical CNN architecture includes:

- **Convolutional layers**: Extract features via convolution.
- **Activation functions** (e.g., ReLU): Introduce non-linearity.
- **Pooling layers** (e.g., max pooling): Downsample to reduce spatial dimensions and computational load.
- **Fully connected layers**: For final classification or regression.
- **Output layer**: Often softmax for classification.

Training involves the **forward pass** (computing predictions) and **backward pass** (backpropagation to update weights via gradients). Below, I'll focus on the math for the convolutional layer, as it's the core of CNNs. I'll assume 2D convolution for images (input shape: batch_size × channels × height × width).


### Forward Pass in a Convolutional Layer

The forward pass computes the output feature map by sliding a kernel over the input.

#### Key Notations:

- Input: \( X \in \mathbb{R}^{B \times C_{in} \times H_{in} \times W_{in}} \) (batch size \( B \), input channels \( C_{in} \), height \( H_{in} \), width \( W_{in} \)).
- Kernel (weights): \( W \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w} \) (output channels \( C_{out} \), kernel height \( K_h \), width \( K_w \)).
- Bias: \( b \in \mathbb{R}^{C_{out}} \) (one per output channel).
- Stride: \( s \) (step size for sliding the kernel).
- Padding: \( p \) (zeros added to borders to control output size).
- Output: \( Y \in \mathbb{R}^{B \times C_{out} \times H_{out} \times W_{out}} \), where \( H_{out} = \lfloor \frac{H_{in} + 2p - K_h}{s} \rfloor + 1 \), similarly for \( W_{out} \).

#### Convolution Operation:

For each position \( (i, j) \) in the output feature map, for batch item \( b \) and output channel \( c_{out} \):

\[
Y_{b, c_{out}, i, j} = \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} W_{c_{out}, c_{in}, m, n} \cdot X_{b, c_{in}, s \cdot i + m - p, s \cdot j + n - p} + b_{c_{out}}
\]

This is essentially a dot product between the kernel and a local patch of the input, summed over input channels, plus bias.

- **How to arrive at this**: The formula derives from cross-correlation (convolution is flipped cross-correlation, but in ML, we often use cross-correlation for simplicity). The indices ensure the kernel slides without going out of bounds (padding handles edges). For valid padding (\( p=0 \)), output shrinks; for same padding, \( p = \lfloor \frac{K-1}{2} \rfloor \) keeps size.

After convolution, apply activation: \( Y' = f(Y) \), e.g., ReLU: \( f(x) = \max(0, x) \).

Pooling (e.g., max pooling) over a window (size \( k \), stride \( s \)) takes the max value in each patch, reducing dimensions.

### Backward Pass in a Convolutional Layer

The backward pass computes gradients for weights, biases, and inputs using chain rule, to minimize loss \( L \) via gradient descent.

#### Key Notations:
- Incoming gradient from next layer: \( \frac{\partial L}{\partial Y} \in \mathbb{R}^{B \times C_{out} \times H_{out} \times W_{out}} \) (gradient w.r.t. output).
- We need:
  - \( \frac{\partial L}{\partial W} \) (weight gradient).
  - \( \frac{\partial L}{\partial b} \) (bias gradient).
  - \( \frac{\partial L}{\partial X} \) (input gradient, passed to previous layer).

#### Bias Gradient:
Simple sum over spatial dimensions and batch:

\[
\frac{\partial L}{\partial b_{c_{out}}} = \sum_{b=0}^{B-1} \sum_{i=0}^{H_{out}-1} \sum_{j=0}^{W_{out}-1} \frac{\partial L}{\partial Y_{b, c_{out}, i, j}}
\]

- **How to arrive at this**: Bias is added to every position, so its gradient is the sum of all output gradients per channel.

#### Weight Gradient:
Correlate input with output gradient:

\[
\frac{\partial L}{\partial W_{c_{out}, c_{in}, m, n}} = \sum_{b=0}^{B-1} \sum_{i=0}^{H_{out}-1} \sum_{j=0}^{W_{out}-1} \frac{\partial L}{\partial Y_{b, c_{out}, i, j}} \cdot X_{b, c_{in}, s \cdot i + m - p, s \cdot j + n - p}
\]

- **How to arrive at this**: By chain rule, \( \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial W} \). Since \( Y \) is linear in \( W \), this is like convolving the input with the output gradient (but actually cross-correlation).

#### Input Gradient:
"Full" convolution of rotated kernel with output gradient (to propagate error back):

First, pad the output gradient if needed. Then:

\[
\frac{\partial L}{\partial X_{b, c_{in}, k, l}} = \sum_{c_{out}=0}^{C_{out}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} \frac{\partial L}{\partial Y_{b, c_{out}, i, j}} \cdot W_{c_{out}, c_{in}, m, n}
\]

Where \( i = \lfloor \frac{k + p - m}{s} \rfloor \), \( j = \lfloor \frac{l + p - n}{s} \rfloor \), and only if the division is integer (i.e., aligns with stride).

- **How to arrive at this**: Chain rule: \( \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial X} \). This requires "deconvolving" or transposing the convolution. In practice, it's implemented as convolving the output gradient with a rotated (180°) kernel, with appropriate padding.

If there's activation, multiply by its derivative (e.g., ReLU derivative: 1 if >0, else 0).

For pooling, backward pass upsamples the gradient (e.g., for max pooling, place gradient only at max position).



### Example



https://colab.research.google.com/drive/1AOONNT2DS0xP6k3thCq8pog1PoJUKTe6#scrollTo=b2tjGQ76GbeD


https://en.wikipedia.org/wiki/Region_Based_Convolutional_Neural_Networks

https://ravjot03.medium.com/detecting-vehicles-in-videos-with-faster-r-cnn-a-step-by-step-guide-932bbabf978b

https://cs231n.github.io/assignments2025/assignment3
https://cs231n.stanford.edu/assignments.html

https://www.kaggle.com/code/christofhenkel/temporal-convolutional-network/notebook



## Additional

### When is a Neural Network Considered "Deep"?

The term "deep" in the context of neural networks refers to the architecture's depth, specifically the number of layers (particularly hidden layers) that enable the network to learn hierarchical and abstract representations of data. There is no universally agreed-upon minimum threshold that strictly divides "shallow" from "deep" neural networks, as it can depend on the context, task, and historical usage in research. However, based on established definitions and expert consensus in machine learning, a neural network is generally considered deep if it has at least two hidden layers (in addition to the input and output layers), resulting in a total of at least four layers overall. This aligns with the concept of a substantial credit assignment path (CAP) depth greater than two, where the CAP represents the chain of transformations from input to output.

<div class="grid cards" markdown>

- **Shallow vs. Deep Networks**

    ---

    - A shallow neural network typically has 0 or 1 hidden layer (e.g., a basic perceptron or multilayer perceptron with one hidden layer). These are sufficient for simple tasks but struggle with complex, hierarchical data patterns.
    - Deep networks, by contrast, stack multiple hidden layers to capture increasingly abstract features (e.g., edges in early layers for image recognition, evolving to objects in deeper layers). The "deep" descriptor emphasizes this multi-layer stacking, often with non-linear activations.

- **Historical and Theoretical Basis**

    ---

    - Early deep learning models, such as those from Geoffrey Hinton's group in the 2000s, featured three hidden layers and were pivotal in reviving interest in deep architectures.
    - Jürgen Schmidhuber's work defines "very deep" learning as having CAP depth >10, but the baseline for "deep" starts at multiple non-linear layers (CAP >2).
    - The universal approximation theorem shows that even a single hidden layer can theoretically approximate any function, but in practice, deeper networks (with fewer neurons per layer) are more efficient for complex tasks, avoiding the need for exponentially more neurons in shallow setups.

- **Common Thresholds from Sources**

    ---

    - Most researchers and textbooks agree on **at least 2 hidden layers** as the minimum for "deep" (total layers: input + 2 hidden + output = 4).
    - Some sources, like IBM's overview, specify **more than 3 total layers** (inclusive of input and output), which equates to at least 1 hidden layer—but this is often critiqued as too low, as it would classify basic MLPs as deep.
    - In practice, modern deep networks (e.g., CNNs or transformers) have dozens or hundreds of layers, but the minimum remains the focus for foundational classification.

- **Why the Ambiguity?**

    ---

    - The term "deep" originated as somewhat informal or "marketing" in early literature but has solidified around the multi-layer criterion.
    - For recurrent neural networks (RNNs), depth can be effectively unlimited due to signal propagation through layers over time, but the static layer count still applies for feedforward cases.
    - In educational or certification contexts (e.g., CFA materials), there's occasional debate between 2 or 3 hidden layers, but evidence leans toward 2 as the practical minimum.

</div>

If you're implementing a network, start with 2–3 hidden layers for "deep" experiments and tune based on your dataset and performance metrics (e.g., via cross-validation). For specific architectures like CNNs, the minimum might vary slightly due to convolutional layers.

---

## Interactive: Convolutional Filter Visualizer

Select a filter and watch it slide over a synthetic image, producing the feature map.

<div id="conv-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;font-family:Inter,sans-serif;color:#e6edf3;">
<div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1rem;align-items:center;">
  <div>
    <label style="color:#8b949e;font-size:.85rem;">Filter: </label>
    <select id="conv-filter" onchange="convDraw()" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:3px 8px;">
      <option value="edge-h">Horizontal Edge</option>
      <option value="edge-v">Vertical Edge</option>
      <option value="blur">Blur (avg)</option>
      <option value="sharpen">Sharpen</option>
      <option value="emboss">Emboss</option>
    </select>
  </div>
  <div>
    <label style="color:#8b949e;font-size:.85rem;">Stride: </label>
    <select id="conv-stride" onchange="convReset()" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:3px 8px;">
      <option value="1">1</option>
      <option value="2">2</option>
    </select>
  </div>
  <button onclick="convAnimate()" id="conv-btn" style="padding:5px 18px;background:#3fb950;color:#0d1117;border:none;border-radius:5px;cursor:pointer;font-weight:bold;">&#9654; Animate</button>
  <button onclick="convReset()" style="padding:5px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">&#8635; Reset</button>
</div>
<canvas id="conv-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div id="conv-formula" style="margin-top:.8rem;font-family:monospace;font-size:.8rem;color:#8b949e;background:#161b22;border-radius:5px;padding:.7rem;"></div>
</div>

<script>
(function(){
  const filters = {
    'edge-h':{kernel:[[-1,-1,-1],[0,0,0],[1,1,1]],label:'Horizontal Edge Detector'},
    'edge-v':{kernel:[[-1,0,1],[-1,0,1],[-1,0,1]],label:'Vertical Edge Detector'},
    'blur':{kernel:[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],label:'Average Blur'},
    'sharpen':{kernel:[[0,-1,0],[-1,5,-1],[0,-1,0]],label:'Sharpen'},
    'emboss':{kernel:[[-2,-1,0],[-1,1,1],[0,1,2]],label:'Emboss'},
  };
  const img=[[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0]];
  const N=8,K=3;
  let animId=null,animPos=0,running=false;
  function getStride(){return +document.getElementById('conv-stride').value;}
  function getFilter(){return filters[document.getElementById('conv-filter').value];}
  function convolve(kernel,stride){const out=[];for(let r=0;r<=N-K;r+=stride){const row=[];for(let c=0;c<=N-K;c+=stride){let v=0;for(let kr=0;kr<K;kr++)for(let kc=0;kc<K;kc++)v+=kernel[kr][kc]*img[r+kr][c+kc];row.push(v);}out.push(row);}return out;}
  function normalize(feat){let mn=Infinity,mx=-Infinity;feat.forEach(r=>r.forEach(v=>{mn=Math.min(mn,v);mx=Math.max(mx,v);}));return feat.map(r=>r.map(v=>mx===mn?0.5:(v-mn)/(mx-mn)));}

  window.convDraw=function(pos){
    const canvas=document.getElementById('conv-canvas');
    const ctx=canvas.getContext('2d');
    const W=canvas.parentElement.offsetWidth-48,H=260;
    canvas.width=W;canvas.height=H;canvas.style.height=H+'px';
    ctx.fillStyle='#161b22';ctx.fillRect(0,0,W,H);
    const stride=getStride(),flt=getFilter(),feat=convolve(flt.kernel,stride),featN=normalize(feat),outSz=feat.length;
    const cellIn=Math.min(26,Math.floor((W*0.35)/N)),cellOut=Math.min(32,Math.floor((W*0.32)/outSz));
    const kArea=K*24+10,inX=8,inY=(H-N*cellIn)/2,kx=inX+N*cellIn+12,ky=(H-K*24)/2,outX=kx+kArea+30,outY=(H-outSz*cellOut)/2;
    for(let r=0;r<N;r++)for(let c=0;c<N;c++){ctx.fillStyle=img[r][c]?'#c9d1d9':'#21262d';ctx.fillRect(inX+c*cellIn,inY+r*cellIn,cellIn-1,cellIn-1);}
    ctx.fillStyle='#484f58';ctx.font='9px Inter,sans-serif';ctx.textAlign='center';ctx.fillText('Input ('+N+'×'+N+')',inX+N*cellIn/2,inY+N*cellIn+14);
    ctx.fillStyle='#8b949e';ctx.font='bold 9px Inter,sans-serif';ctx.textAlign='center';ctx.fillText('Kernel',kx+K*24/2,ky-10);
    for(let r=0;r<K;r++)for(let c=0;c<K;c++){const v=flt.kernel[r][c];ctx.fillStyle=v>0?'#1f3d1f':v<0?'#3d1a1a':'#21262d';ctx.beginPath();ctx.roundRect(kx+c*24,ky+r*24,22,22,3);ctx.fill();ctx.fillStyle=v>0?'#3fb950':v<0?'#ff7b72':'#484f58';ctx.font='bold 9px monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(typeof v==='number'&&v%1!==0?v.toFixed(2):v,kx+c*24+11,ky+r*24+11);}
    const ax=outX-16,ay=H/2;ctx.strokeStyle='#484f58';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(ax-10,ay);ctx.lineTo(ax,ay);ctx.stroke();ctx.fillStyle='#484f58';ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(ax-6,ay-4);ctx.lineTo(ax-6,ay+4);ctx.fill();
    for(let r=0;r<outSz;r++)for(let c=0;c<outSz;c++){const v=featN[r][c],hue=v>0.5?(30+v*60)|0:(220);ctx.fillStyle='hsl('+hue+','+(v>0.5?80:v*60|0)+'%,'+(25+v*35|0)+'%)';ctx.fillRect(outX+c*cellOut,outY+r*cellOut,cellOut-1,cellOut-1);if(cellOut>=16){ctx.fillStyle='rgba(0,0,0,0.6)';ctx.font='7px monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(feat[r][c].toFixed(1),outX+c*cellOut+cellOut/2,outY+r*cellOut+cellOut/2);}}
    ctx.fillStyle='#3fb950';ctx.font='9px Inter,sans-serif';ctx.textAlign='center';ctx.textBaseline='alphabetic';ctx.fillText('Feature Map ('+outSz+'×'+outSz+')',outX+outSz*cellOut/2,outY+outSz*cellOut+14);
    if(pos!==undefined){const rs=Math.floor(pos/outSz),cs=pos%outSz,r0=rs*stride,c0=cs*stride;ctx.strokeStyle='#f0883e';ctx.lineWidth=2;ctx.beginPath();ctx.roundRect(inX+c0*cellIn-1,inY+r0*cellIn-1,K*cellIn+2,K*cellIn+2,3);ctx.stroke();ctx.beginPath();ctx.roundRect(outX+cs*cellOut-1,outY+rs*cellOut-1,cellOut+2,cellOut+2,3);ctx.stroke();}
    document.getElementById('conv-formula').innerHTML='Filter: <span style="color:#3fb950;">'+flt.label+'</span> &nbsp;|&nbsp; Out size: <span style="color:#58a6ff;">('+N+'-'+K+')/'+stride+'+1 = '+outSz+'</span> &nbsp;|&nbsp; Params: <span style="color:#f0883e;">'+K*K+'</span> shared over all '+(N*N)+' positions';
  };
  window.convAnimate=function(){running=!running;document.getElementById('conv-btn').textContent=running?'⏸ Pause':'▶ Animate';document.getElementById('conv-btn').style.background=running?'#d29922':'#3fb950';if(running)animLoop();else cancelAnimationFrame(animId);};
  window.convReset=function(){running=false;animPos=0;cancelAnimationFrame(animId);document.getElementById('conv-btn').textContent='▶ Animate';document.getElementById('conv-btn').style.background='#3fb950';convDraw();};
  function animLoop(){const stride=getStride(),outSz=Math.floor((N-K)/stride)+1;convDraw(animPos);animPos=(animPos+1)%(outSz*outSz);if(running)animId=setTimeout(animLoop,90);}
  convDraw();window.addEventListener('resize',convDraw);
})();
</script>

---

--8<-- "docs/2026.2/classes/convolutional-neural-networks/quiz.md"
