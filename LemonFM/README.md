---
license: apache-2.0
---
<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/67d9504a41d31cc626fcecc8/syYHwLv9Z2s7UfTejhvKm.png"> </img>
</div>



[📚 Paper](https://arxiv.org/abs/2503.19740) - [🤖 GitHub](https://github.com/visurg-ai/LEMON) 



This is the official Hugging Face repository for the paper [LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](https://arxiv.org/abs/2503.19740).

This repository provides open access to the *LemonFM* foundation model. For the *LEMON* dataset and our code, please see our at our GitHub repository at [🤖 Github](https://github.com/visurg-ai/LEMON) .

*LemonFM* is an image foundation model for surgery, it receives an image as input and produces a feature vector of 1536 features as output. 


If you use our dataset, model, or code in your research, please cite our paper:

```
@misc{che2025lemonlargeendoscopicmonocular,
      title={LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings}, 
      author={Chengan Che and Chao Wang and Tom Vercauteren and Sophia Tsoka and Luis C. Garcia-Peraza-Herrera},
      year={2025},
      eprint={2503.19740},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.19740}, 
}
```

Abstract
--------
Traditional open-access datasets focusing on surgical procedures are often limited by their small size, typically consisting of fewer than 100 videos and less than 30 hours of footage, which leads to poor model generalization. To address this constraint, a new dataset called LEMON has been compiled using a novel aggregation pipeline that collects high-resolution videos from online sources. Featuring an extensive collection of over 4K surgical videos totaling 938 hours (85 million frames) of high-quality footage across multiple procedure types, LEMON offers a comprehensive resource surpassing existing alternatives in size and scope, including two novel downstream tasks. To demonstrate the effectiveness of this diverse dataset, we introduce LemonFM, a foundation model pretrained on LEMON using a novel self-supervised augmented knowledge distillation approach. LemonFM consistently outperforms existing surgical foundation models across four downstream tasks and six datasets, achieving significant gains in surgical phase recognition (+9.5pp, +9.4pp, and +8.4pp of Jaccard in AutoLaparo, M2CAI16, and Cholec80), surgical action recognition (+4.4pp of mAP in CholecT50), surgical tool presence detection (+5.3pp and +10.2pp of mAP in Cholec80 and GraSP), and surgical semantic segmentation (+8.3pp of mDice in CholecSeg8k). LEMON and LemonFM will serve as foundational resources for the research community and industry, accelerating progress in developing autonomous robotic surgery systems and ultimately contributing to safer and more accessible surgical care worldwide.

How to run our LemonFM foundation model to extract features from your video frames
----------------------------------------------------------------------------------

   ```python
   import torch
   from PIL import Image
   from model_loader import build_LemonFM

   # Load the pre-trained LemonFM model
   lemonfm = build_LemonFM(pretrained_weights = 'your path to the LemonFM')
   lemonfm.eval()

   # Load the image and convert it to a PyTorch tensor
   img_path = 'path/to/your/image.jpg'
   img = Image.open(img_path)
   img = img.resize((224, 224))
   img_tensor = torch.tensor(np.array(img)).unsqueeze(0).to('cuda')

   # Extract features from the image using the ResNet50 model
   outputs = lemonfm(img_tensor)
   ```
