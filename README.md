# liar_project
This is the repository to work on the github repository for the liar LeWagon project

roBERTa training process
1. training on LIAR Dataset with columns statement (= feature)and labels (= traget) in Google Colab Notebook
- labels defined in 3 classes as stated before
- numeric encoding label2id = {"trustworthy": 0, "questionable": 1, "unreliable": 2}
- result: accuracy of 0.49, f1 recall macro of 0.46

2. re-training on modified LIAR Dataset in Vertex AI setup
- Vertex AI setup:
  - dataset: BCS Bucket liar_model/data
  - model: GCS Bucket liar_model/model
  - Server zone: europe-west 1 (Belgium), GPU Quota = 1
- Speaker and Context from the dataset where added to statement
  - eg: Wisconsin is on pace to double the number of layoffs this year. [SPEAKER: katrina-shankland] [CONTEXT: a news conference],2
- lables unchanged
- data sets saved as csv in GCS
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

3. added politifact dataset to the LIAR Datset and retrained model
- preprocessing of politifact data like before: reducing labels to 3, number encoding as stated before, concatination of statement, speaker and context to one final statement
- split into train/validation/test set with 60/20/20 ratio
- merge of LIAR and politifact dataset (train, valid, test) to about 30.000 statements in total
  - test metrices of the trained roBERTa model:
   "test_loss":  "1.0063432455062866n",
   "test_accuracy":  "0.7254866290704021n",
   "test_recall_macro": "0.7021470571188857n",
   "test_precision_macro": "0.6852322634252183n",
   "test_f1_macro": "0.6904495512671932n",
   "test_runtime": 25.1844,
   "test_samples_per_second": 218.27,
   "test_steps_per_second": 13.659,
   "epoch": 5

4. finetuning of parameters
- no finetuniong needed with an f1 macro of 0.69 to reduce CO2 cost
This is the repository to work on the github repository for the liar LeWagon project.
This README file contains explanations on the followed process to get from the raw data to the end product, sequentially.

# Data Preprocessing
## Text Preprocessing

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

## Cleaning with NLTK

Natural Language Toolkit (NLTK) is an NLP library that rpovides preprocessing and modeling tools for text data. It includes several concepts.

### Tokenizing

Tokenizing means to split a sentence into smaller chunks such as individual words called tokens.

Each statement should be tokenized before being fed into the model.

### Stopwords

Stopwords are words that are used so frequently that they don't carry much information especially for topic modeling. NLTK has a built-in corpus of English stopwords that can be loaded and used.

### Lemmatizing

Lemmatizing is a technique used to find the root of words, in order to group them by their meaning rather than by their exact form.
Recommended to use WordNetLemmatizer.
Useful for topic modeling and sentiment analysis.

## Vectorizing

Machine Learning algorithms cannot process raw text, as it needs to be converted into numbers first.
Vectorizing is the process of converting raw text into a numerical representation.
There are multiple vectorizing techniques, among them:
1. **Bag of words:**
2. **Tf_idf:**
3. **N-grams:**

### **Bag of Words:**
Bag of Words representation is one of the most simple and effective ways to represent text for ML models:
- Simply counts how often each word appears in each document of a corpus.
- The count fo each word becomes a feature.
Can be initialized with "CountVectorizer" and finalized with "get_feature_names_out"

### **Tf-idf Representation:**
Stands for "Term Frequency (tf) & document frequency", and follows the pricniple that
- **Term frequency:** The more often a word appears in a document relative to others, the mroe likely it is that it will be important to this document.
- **Document frequency:** If a word appears in many documents of a corpus, however, it shouldn't be that important to understand a particular document. Words appearing too much in too many palces can be considered irrelevant.
Combining the two concepts above, you can compute a "word weight" of a word "x" in a document "d".

With the TfidfVectorizer, we cna go from raw documents to matrix of tf-idf features.

**Key Parameters:**
- max_df / min_df: Set to ignore terms appearing in more than (max_df) or less than (min_df) percent of the documents. Default values are max_df=1 and min_df=0.
- max_features:By specifying "max_features = k" (k being an integer), the CountVectorizer (or the TfidVectorizer) will build a vociabulary that only considers the top k tokens ordered by term frequency across the corpus.

### **N-grams:**
In Bag of Words representation, it is helpful to add context by counting words together in groups of "N" words.
You can define the "N" range with "ngram_range = (min_n, max_n)":
- ngram_range = (1, 3): Will capture the unigrams, the bigrams and the trigrams.

## (Multinomial) Naive Bayes Algorithm:
The Multinomial Naive Bayes algorithm is a classification algorithm based on Bayes' Theorem in probability theory.
It helps represent the probability of a statement being True or not, if it contains certain words, and also the probability of a true/false statement containing certain words.

### Tuning the Vectorizer and the Naive bayes Algorithm Simultaneously
The parameters of both the Vectorizer and the Naive Bayes model need to be tuned simulataneously.
More on how in the lecture.


## Topic Modeling and Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) is an unsupervised algorithm for finding topics in documents.

**Inputs:**
- Document-term matrix: Documents to be converted using a vectorizer
- Number of topics: Number of topics to be discovered within the documents. Each "topic" consists of a set of unordered words -> bag-of-words format
- Number of iterations: LDA is an unsupervised iterative process.

**Outputs:**
- Topics across different documents/pieces of text:
  - These topics can be interpreted as "non-linear Principal Components" of the documents in the corpus.

Implementation of LDA involves the iteration over a lot of different topics, and the gradual assignation of each word in the document to each topic.

**LDA Under the Hood:**
- Goal of LDA is to find topics across documents.
- It converts the vectorized documents (document_term_matrix) into two matrices:
  - document_topic_mixture
  - topic_word_mixture
IMPORTANT: You can select the number of topics you want to detect in your corpus of documents.
- n_components = 2: Topic 0 and topic 1.
Read more about "Under the Hood" on lecture "Natural Language Processing"
