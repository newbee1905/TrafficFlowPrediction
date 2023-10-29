# Traffic Flow Prediction

Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU, CNN, Prophet).

This is the project for our unit COS30018 to upgrade the original repo: [xiaochus/TrafficFlowPrediction](https://github.com/xiaochus/TrafficFlowPrediction).

## Requirement

- python 3.11
- Tensorflow-gpu 2.13.0
- Keras 2.13.1
- scikit-learn 1.3.1
- networkx 3.1
- requests 2.31.0
- folium 0.14.0
- tk 0.1.0
- fastapi 0.103.2

Please use the requirements.txt file to setup the environment.

## Setting up

```bash
pyton -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

**In case installing from requirements.txt fail, please run:**

```bash
pip install keras scikit-learn numpy pandas tensorflow==2.13 prophet networkx==3.1 requests==2.31.0 folium==0.14.0 fastapi pydantic-core pydantic xlrd uvicorn tk gdown
```

## Train the model

**Run command below to train the model:**

```bash
python train.py --model model_name --epoch 1800
```

You can choose "lstm", "gru", "saes", "cnn" or "prophet" as arguments. The ```.h5``` weight file was saved at model folder. For prophet model, there will be a json file in the model folder.

We are training with epoch 1800 on Google Colab, please use smaller epoch like 30 to test the program on your local machine instead of fully train them.

## Experiment

Data are obtained from the Scats Data system provided by Swinburne for Unit 30018 for Semester 2 2023. This dataset covers traffic flow data for the month of October 2006 and focuses on the top A section.

    device: Tesla V100
    dataset: Scats Data on October 2006
    optimizer: Adam(lr=0.01)
    batch_size: 8192


**Run command below to run the program:**

```
python main.py
```

These are the details for the traffic flow prediction experiment.

| Metrics  | MAE       | MSE          | RMSE       | MAPE         | R2          | Explained variance score |
| -------- |:---------:|:------------:| :---------:| :----------: | :---------: | :----------------------: |
| LSTM     | 13.460    | 423.637      | 20.582     | 21.968%      | 0.9439      | 0.9448                   |
| GRU      | 13.375    | 414.661      | 20.363     | 22.510%      | 0.9451      | 0.9451                   |
| SAEs     | 13.371    | 415.322      | 20.379     | 24.148%      | 0.9450      | 0.9450                   |
| CNN      | 13.492    | 420.323      | 20.502     | 24.169%      | 0.9443      | 0.9443                   |
| Prophet  | 47.603    | 3203.730     | 56.602     | 242.007%     | 0.1041      | 0.1640                   |

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
