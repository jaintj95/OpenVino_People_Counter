


# Project Write-Up

## References:  
1) Intel OpenVino Documentation
2) Udacity Intel Edge AI Nanodegree classroom material  
3) OpenVino Github Repo

## Explaining Custom Layers

The process behind converting custom layers involves first determining the original model framework. Based on the original framework being Tensorflow or Caffe, the steps are slightly different.  

Option 1 (Common) - The custom layer can be registered as extension to the model.  
  
Option 2 (Caffe) - The layer can be registered as Custom and then we can use Caffe to compute the layer's output shape. Note: Caffe needs to be installed on the system for this option.  

Option 2 (Tensorflow) - The unsupported subgraph can be replaced with another subgraph. 

Option 3 (Tensorflow) - The actual computation of the unsupported subgraph can be offloaded to Tensforflow during the inference phase.

The reason that developers should handle custom layers is that they device at edge will fail to perform any inference if the custom layers are not converted. Layers that are not included in a list of known layers are classified as Custom Layers by the Model Optimizer. Handling custom layers will ensure that the device performs as expected. 

## Comparing Model Performance

Note: Since I ended up using a model from the OpenVino Model zoo due to poor performance of converted models, I've highlighted all the comparisons for each individual models in the `Model Research` section below. Please go through the entire write-up until the end. All the required write-up material is present there.

My method(s) to compare models before and after conversion to Intermediate Representations
were  

**Conversion procedure:**  
I converted the model using openvino model optimizer through the following set of commands.  

```
tensorflow model - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
mv ssd_mobilenet_v2_coco_2018_03_29 ssd_model
```  


```
export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer
```

```
python MOD_OPT/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config MOD_OPT/extensions/front/tf/ssd_v2_support.json
```  

    
**Cloud vs Edge deployment**:  
Edge devices may require high initial invesment (due to the cost of hardware and setup). However, over a longer timeframe, cloud computing bill can run into thousands of dollars whereas in case of Edge devices there would no such recurring cost.  
Cloud services also required stable internet connectivity. Edge devices have a huge advantage here because they can be deployed anywhere as opposed to cloud based solutions that can only be deployed in areas with internet availability.

## Assess Model Use Cases

In case of retail scenarios, we can use the app to monitor daily in the store. Based on the counts we can also analyse the times at which footfall is highest during the day. Seasonal fluctations such as high footfall during christmas and other holidays can also be accounted for.  

We can deploy these solutions in areas with endangered wildlife to trigger alarms when intruders or poachers are detected in the region.

In regions with high risk of gang based crimes, we can deploy such solutions to detect a sudden increase in number of people in a region and immediately alert the local police department.  

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:  
1) In poor lighting model's accuracy may fail dramatically or even completely drop close to zero. However, this can be mitigated with good hardware that can process the images from poorly lit regions before passing it to the model.  
2) Natural decrease in model accuracy during conversion or other stages may make the model unusable if the doesn't perform the required task such as detecting risk of crimes as mentioned above in use-cases.  
3) Distorted input from camera due to change in focal length and/or image size will affect the model because the model may fail to make sense of the input and the distored input may not be detected properly by the model.  


## Model Research

In investigating potential people counter models, I tried each of the following three models:
Some steps are common to all models so I am highlighting them below:
1. Download the model using `wget` command
2. Un-tar each tar.gz via `tar -xvf filename.tar.gz`
3. (Optional) Rename extracted directors using `mv` command
4. `export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer`
5. `cd` to the model directory

Common steps:  
- Model 1: **SSD Mobilenet V2**
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments: `python MOD_OPT/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config MOD_OPT/extensions/front/tf/ssd_v2_support.json`
  - Pre-conversion stats:  
Inference time => ~55ms  
Model size => 67 MiB  
  - Post-conversion stats:  
Inference time => ~60ms  
Model size => 65 MiB    

- Model 2: **SSD Inception V2**
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments: `python MOD_OPT/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config MOD_OPT/extensions/front/tf/ssd_v2_support.json`
  - Pre-conversion stats:  
Inference time => ~147ms  
Model size => 98 MiB  
  - Post-conversion stats:  
Inference time => ~155ms  
Model size => 96 MiB    
  
- Model 3: **SSD Mobilenet V1**
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments: `python MOD_OPT/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config MOD_OPT/extensions/front/tf/ssd_support.json`
   - Pre-conversion stats:  
Inference time => ~55ms  
Model size => 28 MiB  
  - Post-conversion stats:  
Inference time => ~60ms  
Model size => 26 MiB    

All the models were insufficient because they failed to perform as per expectations. The inference time was high and even changing the probability threshold to values such as 0.3, 0.5, 0.7 etc. did not help as the lower threshold was creating false positives.

![Undetected Person](Undetected.png)

All the models also failed to detect the person above.  
Note - In case the image is not visible, please open the file `Undetected.png` manually.

## Final Model
As all the other tested models did not produce the expected outputs, I finally decided to use a model from the open vino model zoo.  
I noticed two models for my use case: 
1. [person-detection-retail-0002](https://docs.openvinotoolkit.org/latest/person-detection-retail-0002.html)
2. [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

Seeing that the second model has a higher Avg Precision of 88.62% as compared to the 80.14% of the first one, I decided to use the second model.

### Downloading procedure:  
I followed the instructions in Udacity classroom and I am mentioning these steps below:
1. Download all the pre-requisite libraries and source the openvino installation using the following commands  
```
pip install requests pyyaml -t /usr/local/lib/python3.5/dist-packages && clear && source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

2. Navigate to the directory containing the Model Downloader:

```
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader

```

3. Within there, you'll notice a  `downloader.py`  file, and can use the  `-h`  argument with it to see available arguments,  `--name`  for model name, and  `--precisions`, used when only certain precisions are desired, are the important arguments.

4. Use the following command to download the model
```
sudo ./downloader.py --name person-detection-retail-0013 --precisions FP16 -o /home/workspace
```

### Performing Inference
Steps to perform inference using the model
1. Open a new terminal 

	* Execute the following commands:

			cd webservice/server
			npm install
	* After installation, run:

			cd node-server
			node ./server.js

  2. Open another terminal 
	  * Execute the following commands:  

			cd webservice/ui
			npm install
  
	  * After installation, run:

			npm run dev


3. Open another terminal and run:  
```sudo ffserver -f ./ffmpeg/server.conf```

4. Finally execute the following command in another terminal	
```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

### Discussion
Overall, this model performed far better than any of the other models above. It is better at almost every performance metric.  
Inference time => ~39 ms
Model size => 2MiB
Accuracy was also much better, it detected far less false positives and managed to detected the person that all 3 other models failed to detect.  

