
# Diffusion Model Features

## Plan:

1. frequency-based transformations to analyze how specific features emerge and evolve
  - Objective: quantifying evolution of specific features by analyzing transformations in different frequency bands during the denoising process.
  - how features like textures (mid-frequency), edges (high-frequency), and global shapes (low-frequency) are reconstructed
  - Methodology: Fourier or wavelet transforms to decompose images into frequency components at each denoising step; tracking the progression of feature-specific frequency bands to quantify their emergence or refinement
  - aiming for frequency-domain analysis showing the temporal evolution of feature-specific components

2. metrics that can measure the progression of individual features throughout the denoising process
  - Objective: quantitative metrics to measure how individual features (edges, shapes, textures, and colors) evolve at each step of the denoising process
  - measurable relationships between denoising steps and feature stabilization/emergence.
  - Methodology: similarity metrics (e.g., SSIM, edge density) to quantify structural and visual fidelity at each step; new metrics tailored to feature-specific properties, such as edge sharpness or color consistency.
  - aiming for detailed, step-by-step quantitative profile of feature evolution that can be compared across models and configurations

3. mechanistic feature tracking through activation patterns
  - Objective: quantitative relationship between the activation patterns of diffusion models and the emergence of specific features during denoising.
  - correlate activation magnitudes and regions to specific feature transformations, providing a mechanistic understanding of how features emerge
  - Methodology: getting layer-wise and step-wise activations for feature-specific queries (e.g., "edges," "shapes"); statistical techniques to measure the contribution of each layer and timestep to specific features
  - aiming for mechanistic insights into which layers and steps drive the reconstruction of different features


## 1. frequency-based transformations

involves:
- Access and save images at each denoising step from a diffusion model.
- Frequency Analysis: Decomposing these images using Fourier and wavelet transforms.
- Creating metrics to quantify specific features (edges, textures, shapes).
-  Analyzing how these features evolve over time.
- Creating graphs, heatmaps, and visual outputs for the research paper.
- Comparing models and configurations.


## Structure

denoisefeat/  
├── data/                     # Placeholder for data (on HPC; don't upload raw data to GitHub)  
├── models/                   # Pretrained diffusion models  
├── src/  
│   ├── extract_steps.py      # Extract intermediate denoising steps  
│   ├── frequency_analysis.py # Perform Fourier and wavelet transforms  
│   ├── feature_metrics.py    # Compute texture, edge, and shape metrics  
│   ├── temporal_analysis.py  # Analyze feature evolution  
│   ├── visualization.py      # Create visual outputs  
│   └── utils.py  
├── hpc/                      # - for HPC job submission  
│   ├── dataset_setup.pbs     # Script for dataset download/setup  
│   ├── extraction_job.pbs    # Script for running step extraction  
│   ├── analysis_job.pbs      # Script for frequency/metric computations  
│   └── visualization_job.pbs # Script for creating visualizations  
├── notebooks/                # notebooks for exploratory analysis  
├── requirements.txt          # Dependencies  
└── README.md                 # this file


## don't forget in HPC  
source ~/.bashrc
conda activate hpcenv


## else:  
bash Miniconda3-latest-Linux-x86_64.sh -u  
nano ~/.bashrc for export PATH="/rds/general/user/ifc24/home/miniconda3/bin:$PATH" else echo 'export PATH="/rds/general/user/ifc24/home/miniconda3/bin:$PATH"' >> ~/.bashrc
or
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc

conda create -n hpcenv python=3.10 -y
conda activate hpcenv
