# RESNET_TINY_IMAGENET

# HOW TO RUN:

Each folder has a batch file.

To run the experiment for non distributed training for example

`cd non_distribute` 

`sbatch batch.sh`

To run the experiment of Data Parallel (ddp) for example

` cd dataparallel `

` sbatch dparallel.sh `

To run the experiment of Model Parallel for example

` cd ModelParallel `

`sbatch mparallel.sh`


HARDWARE REQUIREMENTS:

For Data Parallel:

<img width="325" alt="image" src="https://user-images.githubusercontent.com/46345142/168456772-5874f18e-3110-4c6c-bacb-cdefe3f7afee.png">


For Model Parallel:

<img width="320" alt="image" src="https://user-images.githubusercontent.com/46345142/168456753-1b5ea1cf-174a-48ac-bf61-f44b1b53a21e.png">

<img width="1009" alt="image" src="https://user-images.githubusercontent.com/63931061/168895691-be60b7c0-c58b-4bb4-bed1-d8dbedee8915.png">

<img width="1005" alt="image" src="https://user-images.githubusercontent.com/63931061/168895834-cba49c93-b17a-4d5c-b5e9-8f6852291ac4.png">
<img width="1028" alt="image" src="https://user-images.githubusercontent.com/63931061/168895902-af4a6585-d1c9-47bf-9675-9d9a6aa40264.png">
<img width="967" alt="image" src="https://user-images.githubusercontent.com/63931061/168895971-8dc3af59-7f38-415e-88da-fe6bda87b8b3.png">




