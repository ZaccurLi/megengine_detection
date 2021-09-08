# README https://github.com/MegEngine/Models/tree/master/official/vision/detection
export PYTHONPATH=./:$PYTHONPATH;
python3 detection/tools/train.py -n 8 -b 8 -f retinanet_res50_3x_800size_arch.py -d /data/dataset/self