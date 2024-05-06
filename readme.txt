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

(copy results from remote)

scp -i C:/Users/henri/Desktop/githubs/xander/HenrijsKey.pem ubuntu@34.224.166.59:/home/ubuntu/xander/results/phiModelNew4.txt C:/Users/henri/Desktop/githubs/code.zip 




## Remote evaluation
1. Launch ec2 instace
2. connect via ssh
ssh -i HenrijsKey.pem ubuntu@34.224.166.59
3. copy code.zip (seperate terminal)
scp -i C:/Users/henri/Desktop/githubs/xander/HenrijsKey.pem C:/Users/henri/Desktop/githubs/code.zip ubuntu@34.224.166.59:/home/ubuntu
4. install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
...
conda create --name xander python=3.10
conda activate xander

5. install nvidia drivers
https://help.ubuntu.com/community/NvidiaDriversInstallation
sudo ubuntu-drivers install
reboot