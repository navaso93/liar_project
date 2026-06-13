# LIAR Project
This is the repository to work on the github repository for the liar LeWagon project

## What is LIAR?
LIAR is the final group project for the "DataScience & AI" bootcamp at LewWagon in Berlin.
The goal is to be able to predict the trustworthiness of political statements from the US, made by different speakers in different contexts.
This will be done by training several models on the dataset, and labelling statements entered by the user in the end product as more or less trustworthy.

## Source Datasets
This project is using data consisting on political statements, and their metadata, that are classified into 6 labels:
- True
- Mostly-True
- Half-True
- Barely-True
- False
- Pants-on-Fire

The data is extracted from two different datasets (both have the same label):
- LIAR Dataset: https://github.com/tfs4/liar_dataset
- Politifact Dataset: https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset

The statements metadata varies from one dataset to another, but the common features are:
- Speaker: Person or entity having said the statement.
- Context: Usually consists on the channel where the statement was made (social media, press..)

Both datasets contained other metadata that as seen in the following section "Preprocessing" has been disregarded so that models are not excesively trained on unknown values for some of the features.
This metadata consisted on:
- LIAR dataset:
  - id: Unique identifier for each statement
  - party: Party or organization belonging to the speaker
  - subject: Subject addressed by the statement
  - job_title: Job title from the speaker
  - state: State where the statement is attributed to
  - barely_true_counts: N° of 'barely_true' statements made by the speaker
  - false_counts: N° of 'false'  statements made by the speaker
  - half_true_counts: N° of 'half_true' statements made by the speaker
  - mostly_true_counts: N° of 'mostly_true' statements made by the speaker
  - pants_on_fire_counts: N° of 'pants-on-fire' statements made by the speaker

Politifact Dataset:
- statement_date: Date attributed to the statement
- factchecker: Name of the person deciding on the label for that statement
- factcheck_date: Date attributed to the statement being fact-checked
- factcheck_analysis: URL to the analysis of the fact-check

Both datasets are merged, column by column, forming a larger dataset of shape (33927, 4).
The data is then preprocessed in order to be used for training the models.


## Data Preprocessing
### Text Preprocessing

- Uppercase & lowercase
All words can be transformed to lowercase. Recommendation is to do it when using traditional NLP models (Bag of Words, TF-IDF...).
However, modern transformer models (BERT, LLMs, etc...) often keep original casing. This is due to distinsguishing some words e.g. Apple (company VS apple (fruit)).
DECISION: We could import a list with all companies or other categories of uppercase words, and include as exception when lowercasing.

Other possible text processing steps:
- **strip:** Removes all whitespaces at the beginning anfd the end of a string. You can also specify a list of characters to be removed at the beginning and end of a string (RECOMMENDED)
- **replace:** Can be used to replace some words by another.
- **split:** Used to split a text into a lsit, where each word is an item.
- **numbers:** Numbers should be removed during text preprocessing steps.
- **punctuation and symbols:** Punctuation and other symbols should be removed for text preprocessing.

RECOMMENDED: Combine "strip + lowercase + numbers + punctuation/symbols" in one function.

- Removing tags with RegEx: Does likely not apply to this project as no webscrapping has been done, but useful to remove HTML tags form text.Used with 'import re'.

### Cleaning with NLTK

Natural Language Toolkit (NLTK) is an NLP library that rpovides preprocessing and modeling tools for text data. It includes several concepts.

#### Tokenizing

Tokenizing means to split a sentence into smaller chunks such as individual words called tokens.

Each statement should be tokenized before being fed into the model.

#### Stopwords

Stopwords are words that are used so frequently that they don't carry much information especially for topic modeling. NLTK has a built-in corpus of English stopwords that can be loaded and used.

#### Lemmatizing

Lemmatizing is a technique used to find the root of words, in order to group them by their meaning rather than by their exact form.
Recommended to use WordNetLemmatizer.
Useful for topic modeling and sentiment analysis.

### Vectorizing

Machine Learning algorithms cannot process raw text, as it needs to be converted into numbers first.
Vectorizing is the process of converting raw text into a numerical representation.
There are multiple vectorizing techniques, among them:
1. **Bag of words:**
2. **Tf_idf:**
3. **N-grams:**

#### **Bag of Words:**
Bag of Words representation is one of the most simple and effective ways to represent text for ML models:
- Simply counts how often each word appears in each document of a corpus.
- The count fo each word becomes a feature.
Can be initialized with "CountVectorizer" and finalized with "get_feature_names_out"

#### **Tf-idf Representation:**
Stands for "Term Frequency (tf) & document frequency", and follows the pricniple that
- **Term frequency:** The more often a word appears in a document relative to others, the mroe likely it is that it will be important to this document.
- **Document frequency:** If a word appears in many documents of a corpus, however, it shouldn't be that important to understand a particular document. Words appearing too much in too many palces can be considered irrelevant.
Combining the two concepts above, you can compute a "word weight" of a word "x" in a document "d".

With the TfidfVectorizer, we cna go from raw documents to matrix of tf-idf features.

**Key Parameters:**
- max_df / min_df: Set to ignore terms appearing in more than (max_df) or less than (min_df) percent of the documents. Default values are max_df=1 and min_df=0.
- max_features:By specifying "max_features = k" (k being an integer), the CountVectorizer (or the TfidVectorizer) will build a vociabulary that only considers the top k tokens ordered by term frequency across the corpus.

#### **N-grams:**
In Bag of Words representation, it is helpful to add context by counting words together in groups of "N" words.
You can define the "N" range with "ngram_range = (min_n, max_n)":
- ngram_range = (1, 3): Will capture the unigrams, the bigrams and the trigrams.

## Training models
Due to the Natural Language Processing nature of the prjoect, we have decided to work with 3 different opitons as models:
- Naive Bayes Algorithm
- Naive Bayes + XG Boost (as will be explained below)
- roBERTa

### (Multinomial) Naive Bayes Algorithm:
The Multinomial Naive Bayes algorithm is a classification algorithm based on Bayes' Theorem in probability theory.
It helps represent the probability of a statement being True or not, if it contains certain words, and also the probability of a true/false statement containing certain words.

### XGBoost stacked with Naive Bayes
Naive Bayes was first trained on the statement text using TF-IDF features. This model is particularly efficient for text classification and produces class probabilities for each statement.

To leverage the information contained in the statement text within an XGBoost model, we used a stacking approach. Specifically, we extracted the probability assigned by Naive Bayes to the "Unreliable" class and used it as an additional feature for XGBoost.

This approach allows XGBoost to benefit from the semantic information captured by the text model without directly processing the high-dimensional TF-IDF representation. XGBoost was then trained on structured features such as speaker, context, party affiliation, historical speaker statistics, and the Naive Bayes probability.

In practice, the Naive Bayes probability acts as a compressed representation of the statement text, enabling XGBoost to combine textual and structured information efficiently.

### Tuning the Vectorizer, the Naive Bayes algorithm, and the XGBoost with GridSearch
In order to find the best values for the different hyperparameters in both the Naive Bayes algorithm and the XG Boost, we used Grid Search, on the following ranges:

- Naive Bayes:
  - "preprocessor__statement__ngram_range": [(1,1), (1,2), (1, 3)]
  - "preprocessor__statement__min_df": [1, 5]
  - "preprocessor__statement__max_df": [0.9, 0.95]
  - "classifier__alpha": [0.1, 1, 10]

- XGBoost:
  - "classifier__n_estimators": [100, 300, 500],
  - "classifier__max_depth": [3, 6, 10],
  - "classifier__learning_rate": [0.01, 0.1, 0.3],
  - "classifier__subsample": [0.8, 1.0]

The results suggested the following:
- Naives Bayes (best scores):
  - "preprocessor__statement__ngram_range": (1, 3)
  - "preprocessor__statement__min_df": 1
  - "preprocessor__statement__max_df": 0.9
  - "classifier__alpha": 0.1

- XGBoost (best scores):
  - "classifier__n_estimators": 100
  - "classifier__max_depth": 3
  - "classifier__learning_rate": 0.3
  - "classifier__subsample": 0.1

With those hyperparameters, and evaluating both models on the test data, we obtain:
- Naive Bayes:
  - 'accuracy': 0.8482169171824344
  - 'recall_macro': 0.8044057697081352
  - 'f1_macro': 0.8287585844792623

- XGBoost:
  - 'accuracy': 0.7269378131447097
  - 'recall_macro': 0.5880440387918694
  - 'f1_macro': 0.5402090162021965

NB shows better performance than XGBoost, which has several explanations:
- Naive Bayes is very effective for text, which is the main feature in this project.
- XGBoost may add noise from weaker features like speaker or context.
- XGBoost may be underfitting due to its simple hyperparameters.
- Class imbalance may hurt XGBoost, especially for the "Questionable" class.
- The stacking setup should be checked, since XGBoost would normally be expected to match or improve on Naive Bayes.

We leave both Naive Bayes (alone) and both (NB + XGBoost combined) as available options for the end product.

### roBERTa training process
Training on LIAR Dataset with columns statement (= feature)and labels (= traget) in Google Colab Notebook:
- Labels defined in 3 classes as stated before
- Numeric encoding label2id = {"trustworthy": 0, "questionable": 1, "unreliable": 2}
- Result: accuracy of 0.49, f1 recall macro of 0.46

Re-training on modified LIAR Dataset in Vertex AI setup
- Vertex AI setup:
  - Dataset: BCS Bucket liar_model/data
  - Model: GCS Bucket liar_model/model
  - Server zone: europe-west 1 (Belgium), GPU Quota = 1
- Speaker and Context from the dataset where added to statement
  - Example: Wisconsin is on pace to double the number of layoffs this year. [SPEAKER: katrina-shankland] [CONTEXT: a news conference],2
- Lables unchanged
- Data sets saved as csv in GCS
  - Result:
    'eval_loss': 1.7021315097808838,
    'eval_accuracy': 0.4660950896336711,
    'eval_recall_macro': 0.4456767574588188,
    'eval_precision_macro': 0.44797375303712644,
    'eval_f1_macro': 0.43657268069171923,
    'eval_runtime': 6.0383,
    'eval_samples_per_second': 212.476,
    'eval_steps_per_second': 13.414,
    'epoch': 5.0

Added politifact dataset to the LIAR Datset and retrained model
- Preprocessing of politifact data like before: reducing labels to 3, number encoding as stated before, concatination of statement, speaker and context to one final statement
- Split into train/validation/test set with 60/20/20 ratio
- Merge of LIAR and politifact dataset (train, valid, test) to about 30.000 statements in total
  - Test metrics of the trained roBERTa model:
   "test_loss":  "1.0063432455062866n",
   "test_accuracy":  "0.7254866290704021n",
   "test_recall_macro": "0.7021470571188857n",
   "test_precision_macro": "0.6852322634252183n",
   "test_f1_macro": "0.6904495512671932n",
   "test_runtime": 25.1844,
   "test_samples_per_second": 218.27,
   "test_steps_per_second": 13.659,
   "epoch": 5

Finetuning of parameters:
- No finetuniong needed with an f1 macro of 0.69 to reduce CO2 cost

At this stage, all 3 model options are giving satisfactory results for the purpose of the project.

## Adding context to model prediction results
In order to add qualitative information to the results given by the models, we decided to include the following features to the project:
- Embedded vectorized version of the dataset into a "Chroma_db" file, used to retrieve the most similar statements available in it, compared to any predicted statement given by the user.
- An explanation given by an LLM aiming at explaining the label given by the models to the predicted statement, especially in relationship to the retrieved similar statements.

### Embedded dataset and retrieved statements
We have used a Chroma database to embed our dataset and later access it to retrieve statements similar to the one entered by the user and predicted.
Code functions related to this section can be seen in "naive-model_navaso93.ipynb", and also in "vectorstore.py".
Chroma database contains all embeded statements as content, with 'speaker' and 'context' as metadata. The retrieved sentences can be filteredby speaker and context, in case user wants to see simialr statements form a single speaker, or from a single context.

### LLM Explanation
The code relative to the LLM information retrieval can be seen in "gemini.py".
First the prompt is generated, by adding contextual information on the prediction (label, score...) as well as the similar sentences retrieved in the previous step.The LLM is then asked to give an explanation relating both.
The connection to the LLM is then done under defined hyperparameters as seen in function 'generate_gemini_explanation'.


## MLOps Architecture


## Website and final product
The LIAR web application provides an interactive interface for evaluating the trustworthiness of political statements.

Built with Streamlit, the application allows users to submit a statement together with optional contextual information such as the speaker and communication context. The request is sent to a FastAPI backend, which performs the prediction and returns additional analysis.

### Main Features:

- Political statement trustworthiness prediction
- Support for multiple Machine Learning models
- Confidence scores and class probabilities
- Retrieval of similar historical statements using RAG
- AI-generated explanations powered by Gemini
- Interactive dataset insights and statistics

### User Workflow:

- Enter a political statement.
- Optionally provide speaker and context information.
- Select the prediction model.
- Submit the request.
- Review:
  - Predicted label (Trustworthy, Questionable, or Unreliable)
  - Confidence scores
  - Similar statements retrieved from the knowledge base
  - AI-generated explanation


### Application Structure

The interface is organized into the following sections:

- Hero Section
- About LIAR
- Dataset Insights
- Statement Input
- Prediction Results
- Similar Statements (RAG)
- AI Analysis

### Backend Integration

The frontend communicates with a FastAPI service deployed on Google Cloud Run. Predictions, retrieved examples, and explanations are returned as JSON responses and displayed dynamically within the application.

## Technologies
Overview on LIAR#s used technologies:
- Streamlit
- FastAPI
- Google Cloud Run
- Gemini
- ChromaDB
- Scikit-learn
- XGBoost
- RoBERTa
