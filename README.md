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
<img width="392" alt="image" src="https://user-images.githubusercontent.com/46345142/168456740-208dac19-3102-4856-9f03-36b5dc09bd38.png">

For Model Parallel:
<img width="320" alt="image" src="https://user-images.githubusercontent.com/46345142/168456753-1b5ea1cf-174a-48ac-bf61-f44b1b53a21e.png">

