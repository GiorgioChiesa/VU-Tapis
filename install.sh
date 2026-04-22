pip install 'git+https://github.com/facebookresearch/fvcore'
pip install 'git+https://github.com/facebookresearch/fairscale'
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
export 
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# pip uninstall torch torchvision torchaudio -y
cd region_proposals/mask2former/modeling/pixel_decoder/ops
pip install -e .
cd ../../../../..

# clear
git clone https://huggingface.co/visurg/LemonFM

# clear