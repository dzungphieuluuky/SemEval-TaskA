# Network Architecture and Information Flow

The following sequence details the flow of information through the modified `microsoft/unixcoder-base` model, including the newly implemented `MultiSampleDropoutClassifier` and structured loss functions (`FocalLossWithSmoothing`).

```mermaid
flowchart TD
    %% Inputs
    Input[("Raw Code Snippet\n(String)")]
    
    %% Tokenization & Preprocessing
    subgraph Preprocessing
        Augment["CodeAugmenter\n(Optional transforms)"]
        Tokenizer["AutoTokenizer\n(microsoft/unixcoder-base)"]
    end
    
    %% Transformer Backbone
    subgraph Transformer["Transformer Backbone"]
        UniXcoder["UniXcoder Layers\n(12 Hidden Layers)"]
    end
    
    %% Custom Head
    subgraph MultiSampleHead["Multi-Sample Dropout Classifier"]
        Pooling["Pool First Token\n(features[:, 0, :])"]
        InitDrop["Initial Dropout\n(dropouts[0])"]
        Dense["Dense Layer\n(nn.Linear + Tanh)"]
        
        subgraph MSD["Multi-Sample Branches"]
            direction LR
            D1["Dropout (p=0.1)"] --> L1["Linear Proj"]
            D2["Dropout (p=0.2)"] --> L2["Linear Proj"]
            D3["Dropout (p=0.3)"] --> L3["Linear Proj"]
            DK["..."] --> LK["..."]
            D5["Dropout (p=0.5)"] --> L5["Linear Proj"]
        end
        
        Mean["Stack & Mean\n(Average Logits over K passes)"]
    end
    
    %% Outputs & Loss
    subgraph OutputLoss["Loss Computation"]
        Logits["Final Logits\n(Shape: [N, Num_Classes])"]
        Loss["FocalLossWithSmoothing\n(Label Smoothing + Focal Weight)"]
        Labels[("Ground Truth Labels")]
    end

    %% Flow connections
    Input --> Augment
    Augment --> Tokenizer
    Tokenizer -- "input_ids, attention_mask" --> UniXcoder
    
    UniXcoder -- "features [Batch, Seq, Hidden]" --> Pooling
    Pooling -- "[Batch, Hidden]" --> InitDrop
    InitDrop --> Dense
    
    Dense --> D1
    Dense --> D2
    Dense --> D3
    Dense --> DK
    Dense --> D5
    
    L1 --> Mean
    L2 --> Mean
    L3 --> Mean
    LK --> Mean
    L5 --> Mean
    
    Mean --> Logits
    
    Logits --> Loss
    Labels --> Loss
```

### Key Enhancements Visualized:
1. **Multi-Sample Dropout Classifier:** The diagram shows how the pooled hidden state branches into multiple dropout layers. The network performs classification on each branch in parallel and averages the resultant logits, improving generalization without adding parameters.
2. **Pooling:** Unlike standard sequence classification which sometimes uses a designated pooler, the current flow manually extracts the first token representation `features[:, 0, :]` before feeding it into the multi-sample dropout projection.
3. **Robust Loss Computation:** The averaged final logits pass into `FocalLossWithSmoothing`, allowing the model to simultaneously handle class imbalance (Focal Loss) and over-confidence constraints (Label Smoothing).