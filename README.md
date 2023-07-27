## Predicting river water  in Le Thuy, Kien Giang river based on deep learning techniques
#### Authors: Trieu Trung Hieu, Ta Quang Chieu, Nguyen Dac Hieu, Dinh Nhat Quang, Nguyen Thanh Tung

This repository includes data and code for the paper "Predicting river water  in Le Thuy, Kien Giang river based on deep learning techniques", which was published in the 2022 1st international symposium on Integrated Flood and Sediment Management in River Basin for Sustainable Development (FSMaRT) in Da Nang.

### Requirements

Recent versions of TensorFlow, keras and sklearn are required. You can install all the required packages using the following command:

	$ pip install -r requirements.txt

### Running the code

The preprocess.sh script is used to preprocess raw inputs (including historical data of rainfall and water level for the 3 stations Kien Giang, Dong Hoi, Le Thuy. The following parameters are available to create different data scenarios, length of inputs, and number of timesteps ahead of prediction: 
* `--scenario` -- Used to create different data scenarios in our paper (Scenario 1: Only use historical water level data, Scenario 2: Use historical water level and rainfall data, and Scenario 3: Use hitorical water level, rainfall, and average predicting rainfall data)
* `--target` -- Used to specify the target columns. Our paper has experimented with 

#### Input format

Tbd.
