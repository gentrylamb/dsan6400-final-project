# Modeling Movie Success from Collaboration Networks Using Graph Neural Networks

[Google Drive Link](https://drive.google.com/drive/folders/1gqDknYkFSzL_MCEmWMnsA7U3kJiCd9M1?usp=drive_link)

This project explores the prediction of movie success metrics such as box office revenue and critical acclaim using a combination of traditional movie metadata and network analysis of crew collaborations. We construct and analyze social networks from cast and crew data, extracting graph-based features like centrality measures to augment predictive models. Both baseline machine learning models and advanced graph neural networks (GAT, GCN) are implemented to classify movies by success and understand the impact of network relationships in movie outcomes. This work provides insights into how collaboration patterns influence movie performance and serves as a foundation for recommendation systems and industry analytics.

## Team
- Member 1 - David Corcoran
- Member 2 - Gentry Lamb
- Member 3 - Adam Stein

## Repository Structure

```
├── README.md                         # Main project overview and instructions.
├── articles/                         # Research papers and academic references used in the project. 
├── requirements.txt                  # List all the packages and librariesneeded for the project to run.
├── code/                             # Jupyter notebooks for each major step of the project.
    ├── data_collection.ipynb             # Scripts for collecting raw movie and crew data from APIs.  
    ├── data_cleaning.ipynb               # Scripts for cleaning and preparing raw data.  
    ├── eda.ipynb                         # Exploratory Data Analysis: visualizations and summary statistics.  
    ├── network_creation.ipynb            # Creates crew collaboration and movie similarity graphs.  
    ├── graph_attributes.ipynb            # Calculates graph-based features such as centrality measures.  
    ├── supervised_learing.ipynb          # Baseline machine learning classification models.  
    └── GNN_Classification.ipynb          # Graph Neural Network models (GAT & GCN) for classification.
├── data/                             # Data files.
├── assets/                           # Figures and plots generated during analysis and modeling.
└── report/                           # Final technical report and supporting files.
```

## Data

The `data/` folder contains both the original (raw) datasets and cleaned datasets used for modeling.

## Key Findings

- **Network Features Alone May Not Improve Prediction:** Incorporating raw centrality measures (degree, closeness, betweenness) into regression models did not consistently improve prediction of revenue or ratings, likely due to collinearity and noise. Applying dimensionality reduction (PCA) to network features may enhance performance.
- **Graph Neural Networks Show Promise:** GNN models (GAT, GCN) outperformed traditional baselines in classifying movie success, indicating that leveraging graph structure and learned embeddings provides richer predictive signals than tabular features alone.
- **Defining Success is Crucial:** Success classification depends heavily on target definitions (e.g., revenue or rating thresholds). Careful selection of these thresholds affects model outcomes and interpretability.
- **Data Cleaning and Feature Engineering Matter:** Effective preprocessing of categorical metadata (genres, crew roles) and network aggregation to movie-level features are essential for meaningful model input.
