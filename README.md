## Predicting river water  in Le Thuy, Kien Giang river based on deep learning techniques
#### Authors: Trieu Trung Hieu, Ta Quang Chieu, Nguyen Dac Hieu, Dinh Nhat Quang, Nguyen Thanh Tung

This repository includes data and code for the paper "Predicting river water  in Le Thuy, Kien Giang river based on deep learning techniques", which was published in the 2022 1st international symposium on Integrated Flood and Sediment Management in River Basin for Sustainable Development (FSMaRT) in Da Nang.

### Requirements

Recent versions of TensorFlow, keras and sklearn are required. You can install all the required packages using the following command:

	$ pip install -r requirements.txt

### Installing the code

Install the code by running the following commands:

	$ cd /folder/containing/the/project
 	$ pip install -e Water-level-prediction 

When finish running the code, uninstall by running the following command:

 	$ pip uninstall src

### Running the code

The preprocess.sh script is used to preprocess raw inputs (including historical data of rainfall and water level for the 3 stations Kien Giang, Dong Hoi, and Le Thuy. The following parameters are available to create different data scenarios, length of inputs, and number of timesteps ahead of prediction: 
* `--s` -- Different data scenarios in our paper (Scenario 1: Only use historical water level data, Scenario 2: Use historical water level and rainfall data, and Scenario 3: Use hitorical water level, rainfall, and average predicting rainfall data) (Default value: 1)
* `--t` -- Specify the target columns. Our paper has experimented with predicting the water level at 03 stations Kien Giang (H_KienGiang), Dong Hoi (H_DongHoi), and Le Thuy (H_LeThuy) (Default value: "H_LeThuy")
* `--n` -- Specify the number of time lags used for the features (Default value: 3)
* `--l` -- Specify the number of time leads used for the target variable (Default value: 1)

	$ # Example of preprocess.sh run
	$ bash bin/preprocess.sh -s 1 -t "H_LeThuy" -n 3 -l 1
  
The train.sh script is used to train and evaluate 03 types of RNN models (LSTM, Bidirectional LSTM, GRU) against the test set. Evaluation metrics include R^2, RMSE, MAE, and Max Error Value (The difference between the predicted value and the maximum value in the test set, also known as the peak value during the flooding season). The following parameters are available to create different data scenarios, length of inputs, and number of timesteps ahead of prediction: 
* `--p` -- Whether to use pretrained model (Default value: False)
* `--m` -- Specify the type of model to train and predict ("LSTM", "Bi-LSTM", "GRU") (Default value: "LSTM")
* `--n` -- Specify the number of time lags used for the features (Default value: 3)
* `--l` -- Specify the number of time leads used for the target variable (Default value: 1)

	$ # Example of training a new model
	$ bash bin/train.sh -s 1 -t "H_LeThuy" -n 3 -l 1

	$ # Example of loading and predicting with a pretrained model
	$ bash bin/train.sh -m "Bi-LSTM" -n 3 -l 1
