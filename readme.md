# Cyclops NN Models

This repo contains scripts for building NN models. Right now we're only looking
at standard pretrained models like YOLOv8, YOLO11. However, these scripts don't
care about the model, so once we get to the stage where we have enough training
data to create new models, then we'll figure out a strategy for naming those
models, etc.

## Hailo

Use the 10GB docker image. This method actually works!

I'm using the 2017 COCO validation set as calibration images.

1. Run `create-standard-models.go` to create the onnx files.
2. `./hailo_ai_sw_suite_docker_run.sh` to run the 10GB hailo docker image.
3. `cp /home/ben/dev/cyclops/models/yolo*.onnx /home/ben/dev/hailo-docker/shared_with_docker`
4. `rsync -av /home/ben/mldata/coco/val2017/ /home/ben/dev/hailo-docker/shared_with_docker/val2017/`

Inside docker image:

1. `cd ../shared_with_docker`
2. `hailomz compile yolov8n --ckpt=yolov8n.onnx --hw-arch hailo8l --calib-path val2017`
3. And repeat for every model

NOTE: We might want to crank up the optimization settings for the hailomx
compile step.

Outside docker image:

```
mkdir -p /home/ben/dev/cyclops/models/coco/hailo/8L
cp /home/ben/dev/hailo-docker/shared_with_docker/yolov8n.hef /home/ben/dev/cyclops/models/coco/hailo/8L/yolov8n_640_640.hef
cp /home/ben/dev/hailo-docker/shared_with_docker/yolov8s.hef /home/ben/dev/cyclops/models/coco/hailo/8L/yolov8s_640_640.hef
cp /home/ben/dev/hailo-docker/shared_with_docker/yolov8m.hef /home/ben/dev/cyclops/models/coco/hailo/8L/yolov8m_640_640.hef
cp /home/ben/dev/hailo-docker/shared_with_docker/yolov8l.hef /home/ben/dev/cyclops/models/coco/hailo/8L/yolov8l_640_640.hef
```

Each model takes quite a while to build (eg 5 to 10 minutes, I didn't measure).

All of the `compile` commands in one block:

```
hailomz compile yolov8n --ckpt=yolov8n.onnx --hw-arch hailo8l --calib-path val2017
hailomz compile yolov8s --ckpt=yolov8s.onnx --hw-arch hailo8l --calib-path val2017
hailomz compile yolov8m --ckpt=yolov8m.onnx --hw-arch hailo8l --calib-path val2017
hailomz compile yolov8l --ckpt=yolov8l.onnx --hw-arch hailo8l --calib-path val2017
```

## Hailo notes

Take a look at:
`/local/workspace/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/yolov8m.alls`
and compare it to these notes:
https://community.hailo.ai/t/compile-yolov8-onnx-to-hef/2005

This was the final .alls file from the above post (for yolov8n):

```
I added model_optimization_flavor and post_quantization_optimization in .alls file. After that model works fine
Now .alls file looks like:
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
model_optimization_flavor(optimization_level=1, compression_level=0, batch_size=2)
post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, batch_size=8, dataset_size=1112)
nms_postprocess(“/home/hailo/testing/yolov8/yolov8n_nms_config.json”, meta_arch=yolov8, engine=cpu)
```

## Hailo failed attempts at running hailomz

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
