# Terminal and Shell Helper

## 1. Shell basics

Useful commands:

pwd
ls -la
cd
clear

Useful shortcuts:

Ctrl+C: cancel a running command
Ctrl+L: clear the terminal
Ctrl+R: search command history

---

## 2. Pipes and redirects

Pipe:

command1 | command2

Example:

grep "loss" logs/fake_train.log | wc -l

Redirect stdout:

python train.py > output.log

Append stdout:

python train.py >> output.log

Redirect stderr:

python train.py 2> error.log

Redirect stdout and stderr together:

python train.py > train_full.log 2>&1

---

## 3. Log processing

Extract loss values:

awk '{print $4}' logs/fake_train.log > outputs/losses.txt

Watch logs in real time:

tail -f logs/fake_train.log

Filter logs in real time:

tail -f logs/fake_train.log | grep --line-buffered "loss"

---

## 4. tmux

Create a session:

tmux new -s training

Detach:

Ctrl+B, then D

List sessions:

tmux ls

Reattach:

tmux attach -t training

Kill a session:

tmux kill-session -t training

---

## 5. htop

Run:

htop

Quit:

q

Use it to inspect CPU, memory, and running processes.

---

## 6. SSH

Basic remote connection:

ssh user@server-ip

Copy file to remote:

scp file.py user@server-ip:~/

Port forwarding for remote Jupyter:

ssh -L 8888:localhost:8888 user@server-ip
