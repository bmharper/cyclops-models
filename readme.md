# Cyclops NN Models

This repo contains scripts for building NN models. Right now we're only looking
at standard pretrained models like YOLOv8, YOLO11. However, these scripts don't
care about the model, so once we get to the stage where we have enough training
data to create new models, then we'll figure out a strategy for naming those
models, etc.

## Hailo

### How to export models for Hailo accelerators

ehhh. NOPE! This is NOT what we need! It doesn't have `hailomz`, which is what
we need.

1. Download the Hailo Dataflow Compiler
   [from hailo](https://hailo.ai/developer-zone/software-downloads/) The
   filename when I wrote this document was
   `hailo_dataflow_compiler-3.29.0-py3-none-linux_x86_64.whl`
2. Create a python 3.10 environment (eg `conda create --name hailo python=3.10`)
   If the python version has changed for the Dataflow Compiler, then obviously
   use that version of Python that they recommend.
3. `conda activate hailo`
4. `sudo apt install graphviz graphviz-dev`
5. `pip install hailo_dataflow_compiler-3.29.0-py3-none-linux_x86_64.whl`
6. `pip install ultralytics`
7. `go run create-standard-models.go`

Let's try again:

1. `conda create --name hailozoo python=3.10`
2. conda activate hailozoo
3. pip install numpy
4. pip install hailo_model_zoo-2.13.0-py3-none-any.whl
5. NOPE. That also fails with some package that fails to build. Maybe my GCC
   version is out of date or something.

# ATTEMPT 3

Run the 10GB docker image.

1. Run docker image
2. `cp /home/ben/dev/cyclops/models/yolov8n.onnx /home/ben/dev/hailo-ai-sw-suite/shared_with_docker`
3. `rsync -av /home/ben/datasets/coco/val2017/ /home/ben/dev/hailo-ai-sw-suite/shared_with_docker/val2017/`

Inside docker image:

1. `cd ../shared_with_docker`
2. `hailomz compile yolov8n --ckpt=yolov8n.onnx --hw-arch hailo8l --calib-path val2017`

######################################################################################################

## OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD

These are pretrained cyclops NN models.

`bin`/`param` files are for [ncnn](https://github.com/Tencent/ncnn).

## Origins

| Model       | Origin                                                                                  |
| ----------- | --------------------------------------------------------------------------------------- |
| yolov7-tiny | [github.com/nihui/ncnn-assets](https://github.com/nihui/ncnn-assets/tree/master/models) |
| yolov8n     | Exported using YOLOv8 (ultralytics) exporter to ncnn format                             |
| yolov8s     | Exported using YOLOv8 (ultralytics) exporter to ncnn format                             |
