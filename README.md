# Traffic Flow Prediction

Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU, CNN, Prophet).

This is the project for our unit COS30018 to upgrade the original repo: [xiaochus/TrafficFlowPrediction](https://github.com/xiaochus/TrafficFlowPrediction).

## Requirement
- Python 3.11
- Tensorflow-gpu 2.13.0
- Keras 2.13.1
- scikit-learn 1.3.1

Please use the requirements.txt file to setup the environment.

## Setting up

```bash
pyton -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Train the model

**Run command below to train the model:**

```
python train.py --model model_name
```

You can choose "lstm", "gru", "saes", "cnn" or "prophet" as arguments. The ```.h5``` weight file was saved at model folder. For prophet model, there will be a json file in the model folder.


## Experiment

Data are obtained from the Caltrans Performance Measurement System (PeMS). Data are collected in real-time from individual detectors spanning the freeway system across all major metropolitan areas of the State of California.
	
	device: Tesla K80
	dataset: PeMS 5min-interval traffic flow data
	optimizer: RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
	batch_szie: 256 


**Run command below to run the program:**

```
python main.py
```

These are the details for the traffic flow prediction experiment.


| Metrics | MAE    | MSE       | RMSE    | MAPE      | R2       | Explained variance score |
| ------- |:------:| :-------: | :-----: | :-------: | :------: | :----------------------: |
| LSTM    | 13.40  | 419.94    | 20.49   | 22.85%    | 0.9444   | 0.9445                   |
| GRU     | 13.36  | 415.87    | 20.39   | 23.31%    | 0.9449   | 0.9450                   |
| SAEs    | 13.32  | 417.80    | 20.44   | 23.54%    | 0.9446   | 0.9452                   |
| CNN     | 13.52  | 421.14    | 20.52   | 24.61%    | 0.9442   | 0.9445                   |
| Prophet | 81.29  | 9984.90   | 99.92   | 475.04%   | -0.8767  | -0.8360                  |

![evaluate](/images/eva.png)

## Features

- Predict data using listed models
- Has GUI and route calculation system

## Reference

- **Traffic Flow Prediction With Big Data: A Deep Learning Approach**  
  *Y Lv, Y Duan, W Kang, Z Li, FY Wang*,  
  *IEEE Transactions on Intelligent Transportation Systems, 2015, 16(2):865-873*,  
  2015. [^SAEs]

- **Using LSTM and GRU neural network methods for traffic flow prediction**  
  *R Fu, Z Zhang, L Li*,  
  *Chinese Association of Automation, 2017:324-328*,  
  2017. [^RNN]

- **Original GitHub Repository: Traffic Flow Prediction**  
  [TrafficFlowPrediction](https://github.com/xiaochus/TrafficFlowPrediction) by xiaochus.

[^SAEs]: Traffic Flow Prediction With Big Data: A Deep Learning Approach, Y Lv, Y Duan, W Kang, Z Li, FY Wang, IEEE Transactions on Intelligent Transportation Systems, 2015, 16(2):865-873, 2015.

[^RNN]: Using LSTM and GRU neural network methods for traffic flow prediction, R Fu, Z Zhang, L Li, Chinese Association of Automation, 2017:324-328, 2017.


## Copyright
See [LICENSE](LICENSE) for details.
