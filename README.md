<img src="KiDS_SBI_logo.png" width="400" />

# Kilo-Degree Survey - Simulation-Based Inference

This repository supports running a simulation-based/likelihood-free analysis based on forward-simulatios of KiDS-1000 cosmic shear within the CosmoSIS and GLASS frameworks that was used in the following analyses:
- Forward simulations: von Wietersheim-Kramsta et al. in prep.
- Likelihood-free inference: [Lin et al. 2022](https://arxiv.org/abs/2212.04521)

The repository is based on the master branch of KCAP module which can be found [here](https://github.com/KiDS-WL/kcap).

The pipeline is built on CosmoSIS, albeit a [modified version](https://bitbucket.org/tilmantroester/cosmosis/src/kcap/) that is `pip`-installable and doesn't rely on environmental variables.

The KiDS-SBI repository integrates the Generator for Large Scale Structure environment ([Tessore et al. 2023](https://arxiv.org/abs/2302.01942)), which can be found [here](https://github.com/glass-dev/glass), into CosmoSIS.

In addition, the KiDS-SBI implements a new methodology to project 3D power spectra to 2D angular power spectra within the nonLimber module (Reischke et al. in prep.). This is achieved using the Levin method [Levin 1994](https://www.sciencedirect.com/science/article/pii/0377042794001189) and the code is available [here](https://github.com/rreischke/nonLimber_max).

## Installation

Clone the repository:
```
git clone git@github.com:mwiet/kids_sbi.git
cd kids_sbi
```

It's strongly recommended to use some kind of encapsulated environment to install `kids_sbi`, e.g., using `conda`. Here we assume that there is a anaconda installation available, that we need MPI support, and that we're on a machine with up-to-date GCC compilers. Notes on installations on macOS and details on how to set up things manually are [here](#installation-on-macos-and-other-details).

On machines with `module` support (e.g., splinter, cuillin), load the anaconda and openmpi modules first:
```
module load anaconda
module load openmpi
```
If there's no automated way to load these modules, make sure `conda` and MPI executables (`mpif90`) are on your `PATH`. For instructions on how to set up your own conda installation, see [Install conda](#install-conda). If you're using your own anaconda installation, don't load the module as well, as this just causes conflicts.

Now set up the conda environment using the provided `conda_env.yaml` file:
```
conda env create -f conda_env.yaml
```
This creates a `kids_sbi_env` environment that should have all the necessary dependencies. Activate the environment with `source activate kids_sbi_env`. NOTE: GLASS requires python >= 3.9

We need to install CAMB because we use the new python interface for it. If `kcap` is to be used on a local machine, `pip install camb` is all there is to do. On a heterogenous cluster like `splinter` or `cuillin`, we need to build CAMB ourselves, however. To do so, run
```
cd $MAIN_PATH
git clone --recursive git@github.com:cmbant/CAMB.git
cd $MAIN_PATH/CAMB
python setup.py build_cluster
python setup.py install
```

We also need to install GLASS:

```
cd $MAIN_PATH
git clone https://github.com/glass-dev/glass.git $MAIN_PATH/glass
cd $MAIN_PATH/glass
git checkout -b kids_sbi_stable 9af4d02
mv $MAIN_PATH/glass/lensing.py $MAIN_PATH/glass/lensing.py.bak
cp $MAIN_PATH/kids_sbi/glass/lensing.py $MAIN_PATH/glass/lensing.py
pip install -e .
```
Cosmology dependency:
```
cd $MAIN_PATH
git clone https://github.com/glass-dev/cosmology.git $MAIN_PATH/glass_cosmology
cd $MAIN_PATH/glass_cosmology
git checkout -b kids_sbi_stable 4c9052f
pip install -e .
```
CosmoSIS-GLASS interface:
```
cd $MAIN_PATH
git clone https://github.com/mwiet/glass-cosmosis.git $MAIN_PATH/glass-cosmosis
cd $MAIN_PATH/glass-cosmosis
pip install -e .
```

Next, we have to install the nonLimber module:

```
cd $MAIN_PATH
git clone https://github.com/rreischke/nonLimber_matter_shells.git $MAIN_PATH/nonLimber_matter_shells
cd $MAIN_PATH/nonLimber_matter_shells
pip install .
```

Finally, we also have to install [SALMO](https://github.com/Linc-tw/salmo):

```
cd $MAIN_PATH
git clone https://github.com/Linc-tw/salmo.git $MAIN_PATH/salmo
cd $MAIN_PATH/salmo/build
cmake ..
make
```

We can now build kids_sbi (which installs a standalone version of CosmoSIS):
```
cd $MAIN_PATH/kids_sbi
python build.py
```

To uninstall CosmoSIS (for example if you need to get the newest version), run `pip uninstall cosmosis_standalone`. To make a fresh installation of kids_sbi, run `python build.py --clean`.

### Installation on macOS and other details

The default macOS compilers are supported now but `gfortran` still needs to be installed. This can be done with `homebrew` by running `brew install gcc`. Note that `gcc 9.2` seems to be incompatible with the `PolyChord` samplers included in cosmosis, so use a different version (e.g., 9.1).

If no MPI support is required, run `python build.py --no-mpi`.

## Usage

Make sure `conda` and MPI are accessible (e.g., by running `module load anaconda` and `module load openmpi`) and that `kcap_env` is activated (`source activate kcap_env`).
To test that everything is working, run the tests (todo...) and some of the configs in `runs/config`:
```
mkdir runs/output
cosmosis runs/config/KV450_no_sys.ini
```
For MPI:
```
mpirun -n 4 cosmosis --mpi runs/config/KV450_no_sys.ini
```

## Repository structure

N/A
