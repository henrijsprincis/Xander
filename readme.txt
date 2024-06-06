# Repository for Xander
Xander is a language model which uses best first search alongside syntax checking to improve upon the result of only using a language model.
## Setup (Python 3.11 and Ubuntu)
1. Clone this repository
2. Unzip database.zip into a folder in the top level directory of xander
3. Open a terminal and type
4. Create and populate a .env file following .env_example
5. Install requirements. Python 3.10 is recommended (alongside conda)
```
pip install -r requirements.txt
```
6. Download nltk punkt
```python
import nltk
nltk.download('punkt')
```
7. Edit the configuration file -- OPTIONAL
8. Launch main.py
a. Through VSCode Run&Debug (preffered)
```bash
python -m main
```
9. Evaluate the results against gold standard to get results in paper.
```bash
cd eval
python evaluation.py --gold ./dev_gold.sql --pred [PRED_FILE_PATH] --etype all --db ../database --table ../tables.json
```


## 

scp -i C:/Users/henri/Desktop/githubs/xander/HenrijsKey.pem ubuntu@54.160.182.193:/home/ubuntu/xander/results/newLlama5Simple.txt C:/Users/henri/Desktop/githubs/

## Remote evaluation
1. Launch ec2 instace
2. connect via ssh
ssh -i key.pem ubuntu@XX.XXX.XXX.XXX

3. copy code.zip (seperate terminal)
scp -i C:/Users/henri/Desktop/githubs/xander/HenrijsKey.pem C:/Users/henri/Desktop/githubs/xander.zip ubuntu@34.224.97.83:/home/ubuntu
4. install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
...
conda create --name xander python=3.10
conda activate xander

5. install nvidia drivers
https://help.ubuntu.com/community/NvidiaDriversInstallation
sudo ubuntu-drivers install
reboot

6. create a swap file (Additional RAM for loading very large models)
https://askubuntu.com/questions/349156/how-to-use-hard-disk-as-ram-like-in-windows

fallocate -l 10G ~/swapfile
sudo mkswap ~/swapfile -f && sudo swapon -p 1000 ~/swapfile
watch -n 1 free -h
## remove a swap file
sudo swapoff ~/swapfile