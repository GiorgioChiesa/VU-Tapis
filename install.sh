
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install 'git+https://github.com/facebookresearch/fairscale'
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd region_proposals
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
pip install -e .
cd ../../../../..