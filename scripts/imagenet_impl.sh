#execute training in bg
#cmd for # epochs, ration training/validation set to use, epochs >=10,000

# python imagenet_impl.py --train_ratio=.001 --val_ratio=.00025 --epochs=4 --debug=True &
#tmux new-session -s user_imageNet
python imagenet_impl.py --train_ratio=.0025 --val_ratio=.0005 --epochs=1000 --debug=False &
#detach using ctrl+b then d
# tmux attach-session -t user_imageNet