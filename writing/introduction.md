# Introduction

Federated learning approaches address a series of data privacy issues by training models on the data that it has access to. Those models then train a central model by passing each node's weights to the central model. It then iteratively improves on those weights by sending the central model's weights back to the nodes.

Mendelian randomization (MR) has a series of solutions which address the problems solved with federated learning. For example, we can use summary statistics from each individual MR model and conduct meta analysis to understand the differences between two bio-banks that have access to different data. Or we can use two-sample MR to further address gaps in data.

Therefore, we argue that causal applications within federated learning, in regards to mendelian randomization are not appropriate.

Here we are exploring a non-linear federated learning approach, with the goal of identifying pertinent phenotypic predictors of disease. 

where problems that benefit from data originating from multiple databases are necessary but cannot be shared between databases. 