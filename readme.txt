# Repository for Xander
Xander is a language model which uses BFS alongside syntax checking to improve upon the result of only using a languagem odel.

## Setup (Python 3.11 and Ubuntu)
1. Clone this repository
2. Unzip database.zip into a folder in the top level directory of xander
3. Open a terminal and type
```
pip install -r requirements.txt
set -a
. .env
set +a
```

## Run the code
First train a model, then you can demo it with and without syntax checking

```
python main.py
```

## Coming soon

- Code refactoring

- Results replication (evaluation of generator and verifier)