## Requirements
- `numpy`, `pytorch` (v1.7.0+), `cuda 11.1`, `opencv-python`, `matplotlib`, `scipy`.

## Usage
- To run ADNN for simulated tomography, execute `simulation.py`. Default parameters are configured in `config.py`. If you intend to run the reconstruction using real experimental data as described in the paper, refer to the \"Experiment Data\" section below for the data sources.
- Ground truth tomography results for the simulated samples are saved in the `tomography_GT_object` folder.
- Captured intensity images from simulations are saved in the `predict` folder.
- To monitor the optimization loss during the process, use TensorBoard with data located in the `run` folder.
- For reconstructing your own data, adjust the parameters in `config.py` to fit your optical system configuration.

## Results
- Optimization results for each iteration are stored in the `optim_process` folder. You can index these results according to your specific settings.

## Experiment Data
- Data for rat hippocampal neuronal cells is available upon request.
- Data for diatom microalgae and Surirella spiralis diatom can be found in [High-speed in vitro intensity diffraction tomography](https://www.spiedigitallibrary.org/journals/advanced-photonics/volume-1/issue-6/066004/High-speed-in-vitro-intensity-diffraction-tomography/10.1117/1.AP.1.6.066004.full).
- Data for *C. elegans* is available from [Professor Laura Waller's Lab](https://drive.google.com/drive/folders/19eQCMjTtiK8N1f1nGtXlfXkEa8qL6kDl).
