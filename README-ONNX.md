```bash
# 1. Clone the project
git clone https://github.com/nhanerc/recognize-anything.git
cd recognize-anything
git checkout onnx

# 2. Use docker container
docker run -it --rm --gpus all -v `pwd`:/workspace -w /workspace nvcr.io/nvidia/pytorch:24.07-py3
pip install -e .
pip install onnxruntime

# 3. Download pretrained model
mkdir pretrained
wget https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth -P pretrained
wget https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth -P pretrained


# 4. Run inference
python inference_ram_plus.py --image images/demo/demo1.jpg --pretrained pretrained/ram_plus_swin_large_14m.pth
python inference_ram.py --image images/demo/demo1.jpg --pretrained pretrained/ram_swin_large_14m.pth

# 5. Export to ONNX
python onnx/export.py -t ram_plus -p pretrained/ram_plus_swin_large_14m.pth -o pretrained
python onnx/export.py -t ram -p pretrained/ram_swin_large_14m.pth -o pretrained

# 6. Run inference with ONNX
python onnx/inference.py -i images/demo/demo1.jpg -p pretrained/ram_plus.onnx
python onnx/inference.py -i images/demo/demo1.jpg -p pretrained/ram.onnx
```