# yolov5.6_openvino_cpp
yolov5.6.1 OpenVINO的C++实现

适用于最新yolov5.6.1版本

yolov5.6.1 https://github.com/ultralytics/yolov5/tree/v6.1

## 安装 [cmake](https://cmake.org/download/)

CMake 3.22.1(https://cmake.org/files/v3.22/cmake-3.22.1-windows-x86_64.msi)

## 安装[openvino](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)

openvino_DevTools 2021.4.2 LTS(pip install openvino-dev==2021.4.2)

openvino_runtime 2021.4.2 LTS(https://registrationcenter-download.intel.com/akdlm/irc_nas/18320/w_openvino_toolkit_p_2021.4.752.exe)

## 模型转换

openvino支持onnx和IR模型，yolov5.6中的export.py即可以进行转换

其中参数

--weights为指定模型路径

--include为指定输出模型


如下：

```python
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5s.pt', help='model.pt path(s)')
.
.
.
parser.add_argument('--include', nargs='+',
                        default=['onnx','openvino'],
                        help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
```python

另外，如果使用神经网络计算棒NCS2的话,需要将export.py中165行：

（由于NCS2仅支持FP16，因此我们还需要将onnx模型转换为支持FP16的IR模型文件,需要pip安装openvino-dev,版本2021.4.2，和openvino_runtime一致）

```python
cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f}"
```python

改为：

```python
 cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f}  --data_type=FP16"
```python

执行export.py结束后，在weights目录下生成onnx文件和xxx_openvino_model目录

然后就可以愉快地使用了~





