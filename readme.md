# Repository for Xander
Xander is a language model which uses best first search alongside syntax checking to improve upon the result of only using a language model.
## Train Xander (Python 3.10 and Ubuntu)
1. Clone this repository
2. Unzip database.zip into a folder in the top level directory of xander
3. Create and populate a .env file following .env_example
4. Install requirements.
```
pip install -r requirements.txt
```
5. Download nltk punkt
```python
import nltk
nltk.download('punkt')
```
6. Edit the configuration file -- OPTIONAL (You may provide a different model_checkpoint from hugging face or choose whether NormalizedSQL is used)
7. Launch main.py
    - Through VSCode Run&Debug (main)
    - Through terminal
```bash
python -m main
```
8. Evaluate the results against gold standard to get results in paper.
```bash
cd eval
python evaluation.py --gold ./dev_gold.sql --pred [PRED_FILE_PATH] --etype all --db ../database --table ../tables.json
```
NB: When using NormalizedSQL, some queries are filtered => the command becomes
```
python evaluation.py --gold ./dev_gold_normalized_sql.sql --pred [PRED_FILE_PATH] --etype all --db ../database --table ../tables.json
```

## Train Neural Query Checker
1. Complete the procedure for training Xander
2. Clone CodeRL
```bash
git clone https://github.com/salesforce/CodeRL
```
3. Train and evaluate neural query checker
    - Through VSCode (neural query checker)
    - Through terminal
```bash
python -m neural_query_checker.main
```

NB: After training this will give a confusion matrix of Code-t5 small model on the Spider evaluation dataset.