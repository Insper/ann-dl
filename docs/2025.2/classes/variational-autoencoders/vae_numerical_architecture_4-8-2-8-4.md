``` { .mermaid title="VAE Numerical Architecture" }
---
config:
  layout: elk
---
flowchart LR
 subgraph Input["Input"]
    direction TB
        x1["x1"]
        x2["x2"]
        x3["x3"]
        x4["x4"]
  end
 subgraph Hidden1["Hidden1"]
    direction TB
        h11["h11"]
        h12["h12"]
        h13["h13"]
        h14["h14"]
        h15["h15"]
        h16["h16"]
        h17["h17"]
        h18["h18"]
  end
 subgraph Encoder["Encoder"]
    direction LR
        Input
        Hidden1
  end
 subgraph Latent["Latent"]
        l1["l1"]
        l2["l2"]
  end
 subgraph Hidden2["Hidden2"]
    direction TB
        h21["h21"]
        h22["h22"]
        h23["h23"]
        h24["h24"]
        h25["h25"]
        h26["h26"]
        h27["h27"]
        h28["h28"]
  end
 subgraph Output["Output"]
    direction TB
        y1["y1"]
        y2["y2"]
        y3["y3"]
        y4["y4"]
  end
 subgraph Decoder["Decoder"]
    direction LR
        Hidden2
        Output
  end
    x1 --- h11 & h12 & h13 & h14 & h15 & h16 & h17 & h18
    x2 --- h11 & h12 & h13 & h14 & h15 & h16 & h17 & h18
    x3 --- h11 & h12 & h13 & h14 & h15 & h16 & h17 & h18
    x4 --- h11 & h12 & h13 & h14 & h15 & h16 & h17 & h18
    h11 --- l1 & l2
    h12 --- l1 & l2
    h13 --- l1 & l2
    h14 --- l1 & l2
    h15 --- l1 & l2
    h16 --- l1 & l2
    h17 --- l1 & l2
    h18 --- l1 & l2
    l1 --- h21 & h22 & h23 & h24 & h25 & h26 & h27 & h28
    l2 --- h21 & h22 & h23 & h24 & h25 & h26 & h27 & h28
    h21 --- y1 & y2 & y3 & y4
    h22 --- y1 & y2 & y3 & y4
    h23 --- y1 & y2 & y3 & y4
    h24 --- y1 & y2 & y3 & y4
    h25 --- y1 & y2 & y3 & y4
    h26 --- y1 & y2 & y3 & y4
    h27 --- y1 & y2 & y3 & y4
    h28 --- y1 & y2 & y3 & y4
```


A[Input Data] -->|Encode| B[Latent Space];
B -->|Decode| C[Reconstructed Data];
B --> D[Mean (μ)];
B --> E[Log Variance (log σ²)];
D --> F[Sampling (z = μ + σ * ε)];
E --> F;
F --> C;
