pip install 'git+https://github.com/facebookresearch/fvcore'
pip install 'git+https://github.com/facebookresearch/fairscale'
# git clone https://github.com/facebookresearch/detectron2.git
clear 

if [ -d "detectron2" ]; then
  echo "Directory 'detectron2' already exists. Skipping git clone."
else
  git submodule add https://github.com/facebookresearch/detectron2.git detectron2
fi

if [ ! -f "LemonFM/lemonfm.pth" ]; then
  mkdir -p LemonFM
  echo "Downloading LemonFM/lemonfm.pth..."
  curl -L -o LemonFM/lemonfm.pth https://huggingface.co/visurg/LemonFM/resolve/main/lemonfm.pth
else
  echo "LemonFM/lemonfm.pth already exists. Skipping download."
fi

clear
echo All Done
# python -m pip install -e detectron2
# export 
# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# # pip uninstall torch torchvision torchaudio -y
# cd region_proposals/mask2former/modeling/pixel_decoder/ops
# pip install -e .
# cd ../../../../..

# clear
# git clone https://huggingface.co/visurg/LemonFM

# clear