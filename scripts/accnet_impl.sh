#execute training in bg
#cmd for # epochs, ratio training/validation set to use, epochs >=1000

#tmux new-session -s user_accnet
# python accnet_impl.py --debug=True --epochs=2 --load_file
echo "Running accnet_impl scripts"
python accnet_impl.py --train_ratio=.1 --val_ratio=.05 --epochs=750 --debug=True --load_file &

# python accnet_impl.py --train_ratio=.01 --val_ratio=.005 --epochs=1000 --debug=False &
#detach using ctrl+b then d
# tmux attach-session -t user_accnet