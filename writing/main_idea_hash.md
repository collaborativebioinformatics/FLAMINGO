# Main Idea

Federated learning framework in which each participating site trains a local
linear model on its own dataset (datasets differ in sample size). Local model
updates are sent to a central model, which aggregates them into a global
model. The global model is then pushed back to each node to refine/initialize
the next round of local training. This repeats over several communication
rounds until the central model converges.

```mermaid
flowchart TD
    subgraph D1["Node 1"]
        A1[("Dataset 1<br/>n = large")]
        M1["Local Non - Linear Model 1"]
        A1 --> M1
    end

    subgraph D2["Node 2"]
        A2[("Dataset 2<br/>n = medium")]
        M2["Local Non - Linear Model 2"]
        A2 --> M2
    end

    subgraph D3["Node 3"]
        A3[("Dataset 3<br/>n = small")]
        M3["Local Non - Linear Model 3"]
        A3 --> M3
    end

    M1 -- "local params/gradients" --> C["Central Model<br/>(Aggregation)"]
    M2 -- "local params/gradients" --> C
    M3 -- "local params/gradients" --> C

    C --> G{"Convergence<br/>check"}
    G -- "No: broadcast updated<br/>global parameters" --> M1
    G -- "No: broadcast updated<br/>global parameters" --> M2
    G -- "No: broadcast updated<br/>global parameters" --> M3
    G -- "Yes" --> F["Final Global Model"]
```
