# DV-FSR

This is the pytorch implementation 

## Environment
+ Python 3.7.9
+ numpy==1.18.5
+ torch==1.7.0+cu101


## Usage

To run attack DV-FSR

`python main.py --dataset=ML --attack=method_DV-FSR --clients_limit=0.001 --model_type=SASrec2 --device=cuda:0 --lr=0.001 --epochs=30 --agg=common --num_attack=1`

There are two choices on dataset:

`--dataset=ML` and `--dataset=steam`.



## License
The codes are for learning and research purposes only.

