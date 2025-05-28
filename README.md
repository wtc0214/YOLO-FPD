# Enhanced YOLO with Spectral Recalibration for Accurate and Real-Time Sign Language Detection 
Abstract:Sign language serves as a vital communication medium for individuals with hearing impairments, yet conventional convolutional architectures often suffer from significant feature degradation, particularly in high-frequency details and multi-scale feature representation. This paper introduces a novel method, YOLO-FPD, which leverages Fast Fourier Transform (FFT) to construct a dual-domain decoupled feature representation framework. A Parallel Frequency-domain Attention Module (PFMLP) is integrated to dynamically enhance key responses in both frequency and spatial domains, while a Dynamic Heterogeneous Multi-scale Cross-stage Fusion Module (DHMCS-FM) is proposed to improve multi-scale and high-frequency gesture feature capture. Experimental results on public datasets demonstrate that YOLO-FPD achieves state-of-the-art accuracy ( mAP@50 of 93.2\% on the ASL dataset and 92.4\% on the Expression dataset ) while maintaining real-time performance, outperforming several mainstream models. Our approach not only addresses the challenges of high-frequency detail loss and multi-scale feature representation but also establishes a collaborative mechanism between frequency-domain and spatial-domain processing, paving the way for more robust and efficient sign language recognition systems.
# Installation
This implementation is based on YOLOv5, while the former belongs to image dehazing network, and the latter belongs to single-stage target detection network.
python 3.10
pytorch 1.13
torchvision 0.14.1
cuda 11.6

1.Create a new environment
conda create -n name python=3.8
conda activate nam

2.Install dependencies
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

3.Install Ultralytics
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e.

#Training and Prediction
#Train
python train.py model=name.yaml data=data.yaml,epoch=300,batch=8

# Prediction
python detect.py mode=predict model=weight_path  source=dataset_path



