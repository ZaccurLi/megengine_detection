# README https://github.com/MegEngine/Models/tree/master/official/vision/detection
export PYTHONPATH=./:$PYTHONPATH;
# README https://github.com/MegEngine/Models/tree/master/official/vision/detection

export PYTHONPATH=./:$PYTHONPATH;

# train
python3 detection/tools/train.py -n 8 -b 8 -f retinanet_res50_3x_800size_arch.py -d /data/dataset/self


# test
python3 detection/tools/test.py -f retinanet_res50_3x_800size_arch.py -n 8 \
                      -se 35 \
                      -d /data/dataset/self


# inference
python3 detection/tools/inference.py \
  -f retinanet_res50_3x_800size_arch.py  \
  -i /data/dataset/self/arch/arch/1.jpg \
  -w log-of-retinanet_res50_3x_800size_arch/epoch_35.pkl