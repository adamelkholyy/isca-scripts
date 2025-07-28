cd /lustre/projects/Research_Project-T116269/nemo

## load modules
echo Loading modules....

module unload Python/3.11.3-GCCcore-12.3.0                                                   

module use /lustre/shared/easybuild/modules/all
module use PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load Perl/5.36.1-GCCcore-12.3.0


echo Modules loaded.


## execute python script
start_time=$(date +%s)
echo Executing Python script...


python diarize.py -a B314028_301_s10.mp3 --whisper-model large-v3 --language en --no-stem --suppress-numerals


echo Script complete.