# RESNET_TINY_IMAGENET

# HOW TO RUN:

Each folder has a batch file.

To run the experiment for non distributed training for example

`python main.py`

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

![WhatsApp Image 2022-05-17 at 5 44 39 AM](https://user-images.githubusercontent.com/63931061/168851532-9651e8ea-48f5-4049-a0e9-889f49dc5cb9.jpeg)
![WhatsApp Image 2022-05-17 at 5 46 56 AM](https://user-images.githubusercontent.com/63931061/168851615-71992a74-4eaa-41c3-bafa-681e54f5d5f4.jpeg)
![WhatsApp Image 2022-05-17 at 5 48 53 AM](https://user-images.githubusercontent.com/63931061/168851644-7f6a74b6-1e50-4a2e-a201-dfdc53336bdf.jpeg)
