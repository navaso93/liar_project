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
