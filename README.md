# 🚀 NSF Research Awards Abstracts

Welcome to the NSF Research Awards Abstracts project! 🎉 This repository contains my solution for clustering abstracts into topics based on their semantic similarity using unsupervised learning techniques.

## 🗂️ Project Structure (using Cookiecutter 🍪)
    config
    │ ├── model
    │ │ └── model.yaml
    │ └── process
    │ │ └── preprocessing.yaml
    │ └── main.yaml
    data
    │ ├── raw
    │ ├── refined
    │ └── trusted
    docs
    models
    notebooks
    │ └── analysis.ipynb
    src
    │ ├── pycache
    │ ├── mlruns
    │ ├── outputs
    │ ├── main.py
    │ ├── model.py
    │ └── processing.py
    tests
    .gitignore
    .pre-commit-config.yaml
    Makefile
    poetry.lock
    pyproject.toml
    README.md



## 🛠️ Setup

To get started with the project, please follow these steps:

1. Clone this repository:
    ```sh
    git clone https://github.com/Ramseths/nsf_research_awards.git
    ```
2. Install the dependencies:
    ```sh
    poetry install
    ```

## 📊 Data

Using Data Lake Architecture.

The data for this project consists of several paper abstracts provided by the NSF (National Science Foundation). The abstracts are stored in the `data/raw` directory.

## 🚀 Approach

In this project I used a combination of traditional and state-of-the-art NLP techniques to uncover themes in the abstracts. For example the main approach is the use of LDA (Latent Dirichlet Allocation) and on the other hand, combination of Embeddings plus KMeans.

1. **Data Preprocessing**: 
    - Cleaned and preprocessed the text data to remove unnecessary fields.
    - Tokenized the text and removed stopwords.
  
2. **Feature Extraction**:
    - Used TF-IDF and word embeddings to convert text into numerical features.
  
3. **Modeling**:
    - Applied clustering algorithms such as K-Means to group similar abstracts.
    - Utilized topic modeling techniques like LDA (Latent Dirichlet Allocation) for discovering topics.

4. **Evaluation**:
    - Analyzed the resulting clusters and topics to understand their coherence and relevance.

## 📈 Results

The results of clustering and theme modeling can be found in the `notebooks/analysis.ipynb` notebook. Although the results are not expected to be perfect, they provide a better understanding of the abstract themes and show the application of various NLP techniques. In addition, the results are deposited in the refined data layer (simulating Data Lake architecture).

## 🤖 Models

The trained models and their configurations are saved in the `models` directory. You can load and evaluate these models using the provided scripts in the `src` directory.

## 🧪 Run project

To run the main file, use the following command:
```sh
cd src
python main.py
```

Happy clustering!
