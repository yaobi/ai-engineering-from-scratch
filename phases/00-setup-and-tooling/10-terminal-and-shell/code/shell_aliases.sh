# Common aliases for AI engineering work

alias ae='source .venv/bin/activate'
alias gs='git status --short'
alias gl='git log --oneline -10'
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
alias findmodels='find . -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" | xargs du -h 2>/dev/null | sort -rh | head -20'
alias disk='df -h'
alias pywhere='python -c "import sys; print(sys.executable)"'
alias killtraining='pkill -f "python.*train"'
