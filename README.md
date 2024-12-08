# Denoising Feature Interactions

Plan:

denoise-feature-interact/
├── README.md
├── requirements.txt
├── configs/
│   └── base_config.yaml
├── models/
│   └── diffusion_model.py
├── scripts/
│   ├── train.py                # Main training script
│   ├── test.py                 # Testing script
│   └── train_job.pbs           # PBS job submission script (new)
├── utils/
│   ├── feature_extraction.py   # Extract features from model
│   └── visualizations.py       # Plotting and visualization
└── experiments/
    └── results/                # Logs, checkpoints, outputs


don't forget on HPC:
source $HOME/miniconda3/bin/activate
conda create -n hpcenv python=3.10 -y
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate hpcenv
