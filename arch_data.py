#!/usr/bin/env python3
# inflatable arch train dataset
import json
import cv2
import os
import sys
import numpy as np

def create_dir(path):
    if os.path.exists(path):
        print(f"{path} exist")
    else:
        os.makedirs(path)
        print(f"create {path} done")

def create_file(path):
    if os.path.exists(path):
        print(f"{path} exist")
    else:
        with open(path, mode='w', encoding='utf-8') as ff:
            print(f"create {path} done")

def cvt_rect_json(p, w, h):
    return [
        int(p[0][0] * w),
        int(p[0][1] * h),
        int((p[1][0] - p[0][0]) * w),
        int((p[2][1] - p[1][1]) * h)
    ]



def generat_detection_data_json(data_num, data_path, json_path, train_anno_path, test_anno_path):
    print(f"\n[ Function：{sys._getframe().f_code.co_name} ]")
    train_num = int(data_num * 0.9)
    test_num = int(data_num * 0.1)
    print("Original Data")
    print(f"total num : {data_num}")
    print(f"train num : {train_num}")
    print(f"test  num : {test_num}")

    jsonfile_train = {}
    jsonfile_test = {}

    box_num = 1
    p_num = 0
    n_num = 0

    with open(json_path, 'r') as f:
        file = json.load(f)
        images_train = []
        images_test = []
        annotations_train = []
        annotations_test = []
        idx = 0
        for item_id, item in enumerate(file['items']):
            file_name = os.path.join(data_path, item['resources'][0]['s'])
            # print(file_name)
            if item_id < train_num:
                images = images_train
                annotations = annotations_train
            else:
                images = images_test
                annotations = annotations_test

            img = cv2.imread(file_name)
            # print(item["resources"][0]["s"].split("/")[-1])
            if img is None:
                continue
            else:
                image = {}
                image['id'] = idx
                image['width'] = item['resources'][0]['size']['width']
                image['height'] = item['resources'][0]['size']['height']
                image['file_name'] = item["resources"][0]["s"].split("/")[-1]
                images.append(image)
                idx += 1
                if item['results_state'] == 'ok':
                    p_num = p_num + 1
                    for box in item['results']['rects']:
                        annotation = {}
                        annotation['id'] = box_num
                        box_num = box_num + 1
                        annotation['image_id'] = image['id']
                        annotation['category_id'] = 0
                        rect = cvt_rect_json(box['rect'], image['width'], image['height'])
                        if rect[3] < 10 or rect[2] < 10:
                            continue
                        assert len(rect) == 4
                        annotation['bbox'] = rect
                        annotation['area'] = rect[2] * rect[3]
                        annotation['iscrowd'] = 0
                        annotations.append(annotation)
                else:
                    n_num = n_num + 1

    #info
    jsonfile_train['info'] = 'zaccur inflatable arch Dataset for training'
    jsonfile_test['info'] = 'zaccur inflatable arch Dataset for testing'

    #licenses
    license = {}
    license['id'] = 998
    license['name'] = 'a drive license'
    license['url'] = 'www.zaccur.com'
    licenses = []
    licenses.append(license)
    jsonfile_train['licenses'] = licenses
    jsonfile_test['licenses'] = licenses

    #images
    jsonfile_train['images'] = images_train
    jsonfile_test['images'] = images_test

    #annotations
    jsonfile_train['annotations'] = annotations_train
    jsonfile_test['annotations'] = annotations_test

    #categories
    categories = []
    category = {}
    category['id'] = 0
    category['name'] = 'arch'
    category['supercategory'] = 'outdoor'
    categories.append(category)
    jsonfile_train['categories'] = categories
    jsonfile_test['categories'] = categories

    # json dump
    # create_file(train_anno_path)
    with open(train_anno_path, "w") as ff:
        json.dump(jsonfile_train, ff)
    # create_file(test_anno_path)
    with open(test_anno_path, "w") as ff:
        json.dump(jsonfile_test, ff)

    # print('total imgs: ', len(os.listdir('/data/dataset/arch/')))
    print("Detection Data")
    print('total dataset : ', p_num + n_num)
    print('positive data : ', p_num)
    print('negative data : ', n_num)
    print('train data : ', len(images_train))
    print('test  data : ', len(images_test))






# input path
data_path = '/data/dataset/self/arch/'
json_path = os.path.join(data_path, 'job-467099-4-arch.json')

# output path
detection_path = os.path.join(data_path, 'arch_detection')
train_anno_path = os.path.join(detection_path,'train_arch.json')
test_anno_path = os.path.join(detection_path,'test_arch.json')
create_dir(detection_path)

# input num
arch_path = os.path.join(data_path, 'arch')
data_num = len([name for name in os.listdir(arch_path) if os.path.isfile(os.path.join(arch_path, name))])

# processing
generat_detection_data_json(data_num, data_path, json_path, train_anno_path, test_anno_path)