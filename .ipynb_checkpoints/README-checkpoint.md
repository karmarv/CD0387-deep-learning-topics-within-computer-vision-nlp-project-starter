# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
- [x] Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
    - Uploaded to s3://sagemaker-us-east-1-841424068653/dogImages/
    - Screenshot of [dataset-dogimages.png](./dataset-dogimages.png)

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Includes:


- [x] Include a screenshot of completed training jobs
    - HPO tuning jobs  [hpo-tuning-jobs.png](./hpo-tuning-jobs.png)
    - Training Jobs screenshot in [training-jobs.png](./training-jobs.png)
- [x] Logs metrics during the training process
    - Log metrics CSV file [training-job-log-events-viewer-result.csv](./training-job-log-events-viewer-result.csv)
    - Screenshot [training-job-log-events-viewer-result.csv.png](./training-job-log-events-viewer-result.csv.png)
    ```
    INFO:__main__:Running on Device cuda:0
    Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
      0%|          | 0.00/44.7M  [00:00<?, ?B/s]
      ...
      
    100%|██████████| 44.7M/44.7M [00:00<00:00, 62.6MB/s]
    INFO:__main__:4
    INFO:__main__:Start Model Training
    INFO:__main__:Epoch 0,#011 train loss: 14.0000, acc: 0.0000, best loss: 1000000.0000
    INFO:__main__:Epoch 0,#011 valid loss: 7.0000, acc: 2.0000, best loss: 7.0000
    INFO:__main__:Epoch 1,#011 train loss: 9.0000, acc: 1.0000, best loss: 7.0000
    INFO:__main__:Epoch 1,#011 valid loss: 5.0000, acc: 2.0000, best loss: 5.0000
    2022-07-11 17:24:20,758 sagemaker-training-toolkit INFO     Reporting training SUCCESS
    INFO:__main__:Testing Model
    INFO:__main__:Test set: Accuracy: 412/836 = 100%), #011Testing Loss: 7.0
    INFO:__main__:Saving Model
    ```
- [x] Tune at least two hyperparameters
    ```
    learning rate (lr) with a range of .001 - .1
    batch size with options 4, 8, 16, 32
    epochs with a range of 2 to 5
    ```
    - HPO 1 [hpo-tuning-batch8-epoch4.png](./hpo-tuning-batch8-epoch4.png)
    - HPO 2 [hpo-tuning-batch4-epoch-2.png](./hpo-tuning-batch4-epoch-2.png)
- [x] Retrieve the best best hyperparameters from all your training jobs
    - HPO Best tuning Job [hpo-best-job.png](./hpo-best-job.png)
    ```
    {'_tuning_objective_metric': '"average test loss"',
     'batch_size': '"4"',
     'epochs': '2',
     'lr': '0.005374899793978716',
     'sagemaker_container_log_level': '20',
     'sagemaker_estimator_class_name': '"PyTorch"',
     'sagemaker_estimator_module': '"sagemaker.pytorch.estimator"',
     'sagemaker_job_name': '"hpo_tuning-2022-07-11-15-26-34-758"',
     'sagemaker_program': '"hpo.py"',
     'sagemaker_region': '"us-east-1"',
     'sagemaker_submit_directory': '"s3://sagemaker-us-east-1-841424068653/hpo_tuning-2022-07-11-15-26-34-758/source/sourcedir.tar.gz"'}
    ```

## Debugging and Profiling
[profiler-debugging-report.pdf](./profiler-debugging-report.pdf)

**Update**: Give an overview of how you performed model debugging and profiling in Sagemaker

- set the profiler and debugger configuration settings
- trained the model, making sure to embed the debug/profilng hooks within the training code. 
- training code is stored in the train_model.py file and is called from the train_and_deploy notebook.

Output in the notebook execuution of this cell
```
Training jobname: pytorch-training-2022-07-11-17-13-07-784
Region: us-east-1
[2022-07-11 17:30:07.198 datascience-1-0-ml-t3-medium-1abf3407f667f989be9d86559395:20 INFO s3_trial.py:42] Loading trial debug-output at path s3://sagemaker-us-east-1-841424068653/pytorch-training-2022-07-11-17-13-07-784/debug-output
[2022-07-11 17:30:07.732 datascience-1-0-ml-t3-medium-1abf3407f667f989be9d86559395:20 WARNING s3handler.py:183] Encountered the exception An error occurred while reading from response stream: ('Connection broken: IncompleteRead(0 bytes read, 1971 more expected)', IncompleteRead(0 bytes read, 1971 more expected)) while reading s3://sagemaker-us-east-1-841424068653/pytorch-training-2022-07-11-17-13-07-784/debug-output/index/000000000/000000000610_worker_0.json . Will retry now
[2022-07-11 17:30:17.625 datascience-1-0-ml-t3-medium-1abf3407f667f989be9d86559395:20 INFO trial.py:198] Training has ended, will refresh one final time in 1 sec.
[2022-07-11 17:30:18.650 datascience-1-0-ml-t3-medium-1abf3407f667f989be9d86559395:20 INFO trial.py:210] Loaded all steps
['CrossEntropyLoss_output_0', 'gradient/ResNet_fc.0.bias', 'gradient/ResNet_fc.0.weight', 'gradient/ResNet_fc.2.bias', 'gradient/ResNet_fc.2.weight', 'layer1.0.relu_input_0', 'layer1.0.relu_input_1', 'layer1.1.relu_input_0', 'layer1.1.relu_input_1', 'layer2.0.relu_input_0', 'layer2.0.relu_input_1', 'layer2.1.relu_input_0', 'layer2.1.relu_input_1', 'layer3.0.relu_input_0', 'layer3.0.relu_input_1', 'layer3.1.relu_input_0', 'layer3.1.relu_input_1', 'layer4.0.relu_input_0', 'layer4.0.relu_input_1', 'layer4.1.relu_input_0', 'layer4.1.relu_input_1', 'relu_input_0']
334
627
ProfilerConfig:{'S3OutputPath': 's3://sagemaker-us-east-1-841424068653/', 'ProfilingIntervalInMilliseconds': 500, 'ProfilingParameters': {'DataloaderProfilingConfig': '{"StartStep": 0, "NumSteps": 1, "MetricsRegex": ".*", }', 'DetailedProfilingConfig': '{"StartStep": 0, "NumSteps": 1, }', 'FileOpenFailThreshold': '50', 'HorovodProfilingConfig': '{"StartStep": 0, "NumSteps": 1, }', 'LocalPath': '/opt/ml/output/profiler', 'PythonProfilingConfig': '{"StartStep": 0, "NumSteps": 1, "ProfilerName": "cprofile", "cProfileTimer": "total_time", }', 'RotateFileCloseIntervalInSeconds': '60', 'RotateMaxFileSizeInBytes': '10485760', 'SMDataParallelProfilingConfig': '{"StartStep": 0, "NumSteps": 1, }'}}
s3 path:s3://sagemaker-us-east-1-841424068653/pytorch-training-2022-07-11-17-13-07-784/profiler-output


Profiler data from system is available
[2022-07-11 17:30:19.007 datascience-1-0-ml-t3-medium-1abf3407f667f989be9d86559395:20 INFO metrics_reader_base.py:134] Getting 10 event files
select events:['total']
select dimensions:['CPU', 'GPU']
filtered_events:{'total'}
filtered_dimensions:{'GPUMemoryUtilization-nodeid:algo-1', 'CPUUtilization-nodeid:algo-1', 'GPUUtilization-nodeid:algo-1'}
You will find the profiler report in s3://sagemaker-us-east-1-841424068653/pytorch-training-2022-07-11-17-13-07-784/rule-output
```

### Results
**Update**: What are the results/insights did you get by profiling/debugging your model?
- One Major insights in the report was based on the Batch size hyper parameter. This allowed me to test with multiple batch size. Ignoring others as they were more from dataloading and infrastructure perspective.
    ```
    Batch size: 
    The BatchSize rule helps to detect if GPU is underutilized because of the batch size being too small. To detect this the rule analyzes the GPU memory footprint, CPU and GPU utilization. The rule checked if the 95th percentile of CPU utilization is below cpu_threshold_p95 of 70%, the 95th percentile of GPU utilization is below gpu_threshold_p95 of 70% and the 95th percentile of memory footprint below gpu_memory_threshold_p95 of 70%. In your training job this happened 1 times. The rule skipped the first 1000 datapoints. The rule computed the percentiles over window size of 500 continuous datapoints. The rule analysed 1079 datapoints and triggered 1 times.

    Your training job is underutilizing the instance. You may want to consider either switch to a smaller instance type or to increase the batch size. The last time the BatchSize rule triggered in your training job was on 07/11/2022 at 17:15:00. The following boxplots are a snapshot from the timestamps. They the total CPU utilization, the GPU utilization, and the GPU memory usage per GPU (without outliers).
    ```

**Update** Remember to provide the profiler html/pdf file in your submission.
- [x] Included [profiler-debugging-report.html](./profiler-debugging-report.html) file for details


## Model Deployment
**Update**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
- [x] Query endpoint steps [endpoint-deployment-steps.png](./endpoint-deployment-steps.png)

**Update** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
- [x] Deployed endpoint screenshot [endpoint-deployed.png](./endpoint-deployed.png)

## Standout Suggestions
**None**
