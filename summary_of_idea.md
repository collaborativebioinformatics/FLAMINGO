# Federated Mendelian randomisation across multiple biobanks

Mendelian randomisation models in a federated manner.

- Non-linear Mendelian randomisation models for each local node
- Federated learning on the shared models

## Setting

| Site | SNPs `G` | Exposure `X` | Outcome `Y` | Confounders `C` |
|---|---|---|---|---|
| Biobank A | yes | yes | yes | **yes** |
| Biobank B | yes | yes | yes | no |

Causal diagram (the same at both sites):

```mermaid
flowchart LR
    G((G)) --> X((X)) --> Y((Y))
    C((C)) --> X
    C --> Y
```

`G` are the SNPs, `X` the exposure, `Y` the outcome and `C` the confounders of `X` and `Y`.
The instrument is `S = G·w`; it stands in `G`'s place, so the design needs `S ⊥ C`.

Biobank A uses `C` to build a polygenic risk score (PRS) `S = G·w` that predicts `X`
but is (approximately) independent of `C`. Biobank B cannot check that independence
itself, so it relies on the certificate issued by A. Only the weights `w`, the
certificate, and model parameters ever leave a site; individual-level data stays put.

## Flow chart

```mermaid
flowchart TB
    subgraph A["Biobank A  (G, X, Y, C observed)"]
        A1["Fit PRS weights w<br/>maximise assoc(G·w, X)<br/>subject to G·w ⊥ C"]
        A2{"Test G·w ⊥ C<br/>on held-out data"}
        A3["Certify w<br/>(test statistic, p-value, sample size)"]
        A4["Compute S = G·w locally<br/>Fit local MR model Y ~ f(X), instrument S"]
        A1 --> A2
        A2 -- "fails: increase penalty / drop SNPs" --> A1
        A2 -- "passes" --> A3
        A3 --> A4
    end

    subgraph S["Coordinator  (no individual-level data)"]
        S1["Store w + certificate"]
        S2["Aggregate MR model parameters<br/>(e.g. FedAvg over sites)"]
        S3["Pooled causal effect of X on Y"]
        S1 --> S2 --> S3
    end

    subgraph B["Biobank B  (G, X, Y observed, C unobserved)"]
        B1["Receive w + certificate<br/>(never sees C)"]
        B2["Compute S = G·w locally"]
        B3["Fit local MR model Y ~ f(X), instrument S"]
        B1 --> B2 --> B3
    end

    A3 -- "w + certificate" --> S1
    S1 -- "w + certificate" --> B1
    A4 -- "parameters / gradients" --> S2
    B3 -- "parameters / gradients" --> S2
    S2 -. "updated global model" .-> A4
    S2 -. "updated global model" .-> B3
```

## What crosses the network

| From | To | Payload |
|---|---|---|
| A | coordinator | PRS weights `w`, independence certificate |
| coordinator | B | PRS weights `w`, independence certificate |
| A, B | coordinator | local MR model parameters or gradients, each round |
| coordinator | A, B | aggregated MR model, each round |

Never: raw `G`, `X`, `Y`, or `C` from any site.

## Assumptions to state

1. **Transportability of the certificate.** `S ⊥ C` is tested in A's population. For it
   to hold in B, the joint distribution of `(G, C)` must be the same (or close enough)
   across biobanks, e.g. same ancestry and comparable recruitment.
2. **Relevance.** `S` predicts `X` in both biobanks, not just in A. Can be checked locally
   at B without confounders.
3. **Exclusion restriction.** `S` affects `Y` only through `X`; the independence step
   addresses confounding but not horizontal pleiotropy.

## Possible extension

The "predicts `X`" part of the PRS fit only needs `(G, X)`, which both sites have. That
step could itself be federated (both sites contribute gradients of the prediction loss),
while the independence penalty is contributed by A alone. The chart above keeps the PRS
fit entirely at A for simplicity.


![alt text](whiteboard_pictures/flowchart.jpeg)
