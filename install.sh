# Install Ollama (no sudo required)
# if [ ! -d "$HOME/ollama" ]; then
#     echo "Installing Ollama..."
#     mkdir -p ~/ollama
#     cd ~/ollama
    
#     # Download Ollama binary
#     curl -fsSL -o ollama-linux-amd64.tar.zst https://github.com/ollama/ollama/releases/download/v0.21.2/ollama-linux-amd64.tar.zst
    
#     # Install zstandard if not present
#     pip install zstandard --quiet 2>/dev/null || true
    
#     # Extract the archive
#     python3 -c "
# import zstandard
# import tarfile
# with open('ollama-linux-amd64.tar.zst', 'rb') as fh:
#     dctx = zstandard.ZstdDecompressor()
#     with dctx.stream_reader(fh) as reader:
#         with tarfile.open(fileobj=reader, mode='r|') as tar:
#             tar.extractall()
# "
    
#     echo "Ollama installed to ~/ollama/bin/"
#     # Export PATH for Ollama
#     export PATH="$HOME/ollama/bin:$PATH"
#     echo "Ollama PATH exported: $HOME/ollama/bin"
# else
#     echo "Ollama already installed. Skipping."
# fi


uv pip install -r requirements.txt
uv pip install 'git+https://github.com/facebookresearch/fvcore'
uv pip install 'git+https://github.com/facebookresearch/fairscale'
# git clone https://github.com/facebookresearch/detectron2.git
clear 

if [ -d "detectron2" ]; then
  echo "Directory 'detectron2' already exists. Skipping git clone."
else
  git submodule add https://github.com/facebookresearch/detectron2.git detectron2
fi
uv pip install -e detectron2 --no-build-isolation

if [ ! -f "LemonFM/lemonfm.pth" ]; then
  mkdir -p LemonFM
  echo "Downloading LemonFM/lemonfm.pth..."
  curl -L -o LemonFM/lemonfm.pth https://huggingface.co/visurg/LemonFM/resolve/main/lemonfm.pth
else
  echo "LemonFM/lemonfm.pth already exists. Skipping download."
fi

if [ -f ".export_vars.txt" ]; then
  source .export_vars.txt
  wandb login --relogin $WANDB_API
else
  echo ".export_vars.txt not exists. log into wandb manually"
fi 

# python -m pip install -e detectron2
# export 
# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# pip uninstall torch torchvision torchaudio -y
# cd region_proposals/mask2former/modeling/pixel_decoder/ops
# uv pip install -e region_proposals/mask2former/modeling/pixel_decoder/ops --no-build-isolation
# cd ../../../../..

# clear
# git clone https://huggingface.co/visurg/LemonFM

clear
echo All Done