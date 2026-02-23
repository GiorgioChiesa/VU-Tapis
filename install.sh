
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install 'git+https://github.com/facebookresearch/fairscale'
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd region_proposals/mask2former/modeling/pixel_decoder/ops
pip install -e .
cd ../../../../..

clear
