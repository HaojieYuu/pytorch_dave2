# run test
python train.py -r ./driving_dataset/driving_dataset -p ./pytorch_dave2/results/model_l0001.pth -nw 10 -n 50 -l 0.001
python train.py -r ./driving_dataset/driving_dataset -p ./pytorch_dave2/results/model_l00005.pth -nw 10 -n 50 -l 0.0005
python train.py -r ./driving_dataset/driving_dataset -p ./pytorch_dave2/results/model_w0.pth -nw 10 -n 50 -w 0
python train.py -r ./driving_dataset/driving_dataset -p ./pytorch_dave2/results/model_w00005.pth -nw 10 -n 50 -w 0.0005