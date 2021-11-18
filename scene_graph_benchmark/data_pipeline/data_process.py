# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import os
import os.path as op
import json
import cv2
import base64
import argparse
import yaml
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm

import torch
from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader, tsv_writer
from maskrcnn_benchmark.structures.tsv_file_ops import generate_linelist_file
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from maskrcnn_benchmark.engine.inference import inference
from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#|     Format captions.txt: image,caption,split ngăn bởi (,)    |
#|                  File caption_flickr8k.json                  |
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
def build_mini_tsv(cfg):
  assert cfg.CAP_DIR.endswith(".json") or cfg.CAP_DIR.endswith(".txt")
  print("Encoding images and generating mini-tsv files...")
  img_phase_dict = {}
  if cfg.CAP_DIR.endswith(".txt"):
    df_cap = pd.read_csv(cfg.CAP_DIR, sep=',')
    for idx in range(0, df_cap.shape[0], 5):
      img_phase_dict[df_cap["image"].iloc[i]]=df_cap["split"].iloc[i]
  else:
    fp = open(cfg.CAP_DIR, "r")
    captions = json.load(fp)
    captions = captions["images"]
    for cap in captions:
      img_phase_dict[cap["filename"]] = cap["split"]

  def sub_build_mini_tsv(phase):
    tsv_file = os.path.join(cfg.DATA_DIR, phase+".tsv")
    label_file = os.path.join(cfg.DATA_DIR, phase+".label.tsv")
    hw_file = os.path.join(cfg.DATA_DIR, phase+".hw.tsv")
    linelist_file = os.path.join(cfg.DATA_DIR, phase+".linelist.tsv")

    rows = []
    rows_label = []
    rows_hw = []

    for key, value in img_phase_dict.items():
      if value == phase:
        img_key = key.split('.')[0]
        img_path = op.join(cfg.IMG_DIR, key)
        img = cv2.imread(img_path)
        img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])

        # Here is just a toy example of labels.
        # The real labels can be generated from the annotation files
        # given by each dataset. The label is a list of dictionary 
        # where each box with at least "rect" (xyxy mode) and "class"
        # fields. It can have any other fields given by the dataset.
        labels = []
        labels.append({"rect": [1, 1, 30, 40], "class": "Dog"})
        labels.append({"rect": [2, 3, 100, 100], "class": "Cat"})

        row = [img_key, img_encoded_str]
        row_label = [img_key, json.dumps(labels)]
        height = img.shape[0]
        width = img.shape[1]
        row_hw = [img_key, json.dumps([{"height":height, "width":width}])]

        rows.append(row)
        rows_label.append(row_label)
        rows_hw.append(row_hw)
    
    tsv_writer(rows, tsv_file)
    tsv_writer(rows_label, label_file)
    tsv_writer(rows_hw, hw_file)

    # generate linelist file
    generate_linelist_file(label_file, save_file=linelist_file)

  def dump_yaml(phase):
    yaml_dict = {"img": phase+".tsv",
              "label": phase+".label.tsv",
              "hw": phase+".hw.tsv",
              "linelist": phase+".linelist.tsv",
              "labelmap": cfg.DATASETS.LABELMAP_FILE}

    with open(op.join(cfg.DATA_DIR, phase+'.yaml'), 'w') as file:
        yaml.dump(yaml_dict, file)
  
  phase_list = ["train", "val", "test"]
  for p in phase_list:
    sub_build_mini_tsv(p)
    dump_yaml(p)

  print("DONE!")
  
def run_test(cfg, model, distributed, model_name):
    print("Predicting to get prediction.tsv(which have features and labels)...")
    if distributed and hasattr(model, 'module'):
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        if len(dataset_names) == 1:
            output_folder = os.path.join(
                cfg.OUTPUT_DIR, "inference",
                os.path.splitext(model_name)[0]
            )
            mkdir(output_folder)
            output_folders = [output_folder]
        else:
            for idx, dataset_name in enumerate(dataset_names):
                # dataset_name1 = dataset_name.replace('/', '_')
                dataset_name1 = dataset_name.split('/')[-1].split('.')[0]
                output_folder = os.path.join(
                    cfg.OUTPUT_DIR, "inference",
                    dataset_name1,
                    os.path.splitext(model_name)[0]
                )
                mkdir(output_folder)
                output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    labelmap_file = config_dataset_file(cfg.DATA_DIR, cfg.DATASETS.LABELMAP_FILE)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        results = inference(
            model,
            cfg,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            skip_performance_eval=cfg.TEST.SKIP_PERFORMANCE_EVAL,
            labelmap_file=labelmap_file,
            save_predictions=cfg.TEST.SAVE_PREDICTIONS,
        )

        # renaming box_proposals metric to rpn_proposals if RPN_ONLY is True
        if results and 'box_proposal' in results and cfg.MODEL.RPN_ONLY:
            results['rpn_proposal'] = results.pop('box_proposal')

        if results and output_folder:
            results_path = os.path.join(output_folder, "results.json")
            # checking if this file already exists and only updating tasks
            # that are already present. This is useful for including
            # e.g. RPN_ONLY metrics
            if os.path.isfile(results_path):
                with open(results_path, 'rt') as fin:
                    old_results = json.load(fin)
                old_results.update(results)
                results = old_results
            with open(results_path, 'wt') as fout:
                json.dump(results, fout)

        synchronize()

    # evaluate attribute detection
    if not cfg.MODEL.RPN_ONLY and cfg.MODEL.ATTRIBUTE_ON and (not cfg.TEST.SKIP_PERFORMANCE_EVAL):
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        for output_folder, dataset_name, data_loader_val in zip(
            output_folders, dataset_names, data_loaders_val
        ):
            results_attr = inference(
                model,
                cfg,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                skip_performance_eval=cfg.TEST.SKIP_PERFORMANCE_EVAL,
                labelmap_file=labelmap_file,
                save_predictions=cfg.TEST.SAVE_PREDICTIONS,
                eval_attributes=True,
            )

            if results_attr and output_folder:
                results_path = os.path.join(output_folder, "results.json")
                # checking if this file already exists and only updating tasks
                # that are already present. This is useful for including
                # e.g. RPN_ONLY metrics
                if os.path.isfile(results_path):
                    with open(results_path, 'rt') as fin:
                        old_results = json.load(fin)
                    old_results.update(results_attr)
                    results_attr = old_results
                with open(results_path, 'wt') as fout:
                    json.dump(results_attr, fout)

            synchronize()
    print("DONE!")

#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#|Các hàm hỗ trợ cho get_feature_tsv() => Lấy features và labels|
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
def generate_additional_features(rect,h,w):
    mask = np.array([w,h,w,h],dtype=np.float32)
    rect = np.clip(rect/mask,0,1)
    res = np.hstack((rect,[rect[3]-rect[1], rect[2]-rect[0]]))
    return res.astype(np.float32)

def generate_features(x, hw_df):
    idx, data,num_boxes = x[0],x[1],len(x[1])
    h,w,features_arr = hw_df.loc[idx,1][0]['height'],hw_df.loc[idx,1][0]['width'],[]

    for i in range(num_boxes):
        features = np.frombuffer(base64.b64decode(data[i]['feature']),np.float32)
        pos_feat = generate_additional_features(data[i]['rect'],h,w)
        x = np.hstack((features,pos_feat))
        features_arr.append(x.astype(np.float32))
        
    features = np.vstack(tuple(features_arr))
    features = base64.b64encode(features).decode("utf-8")
    return {"features":features, "num_boxes":num_boxes}

def generate_labels(x):
    data = x[1]
    res = [{"class":el['class'].capitalize(),"conf":el['conf'], "rect": el['rect']} for el in data] 
    return res
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#|                                                              |
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|

def get_feature_tsv(cfg, distributed, model_name):
  print("Creating features and labels...")
  data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
  dataset_names = cfg.DATASETS.TEST

  def sub_get_feature(dataset_name, data_loader):

    def process(df, label_index=0, feature_index=0, mode='w'):
      df[1] = df[1].apply(lambda x: x['objects'])

      df['feature'] = df.apply(generate_features, axis=1, hw_df=hw_df)
      df['feature'] = df['feature'].apply(json.dumps)

      df['label'] = df.apply(generate_labels, axis=1)
      df['label'] = df['label'].apply(json.dumps)


      label_file = os.path.join(cfg.USED_DATA_DIR, name+'.label.tsv')
      feature_file = os.path.join(cfg.USED_DATA_DIR, name+'.feature.tsv')
      if not os.path.exists(cfg.USED_DATA_DIR):
        os.makedirs(cfg.USED_DATA_DIR)
        print(f"path to {cfg.USED_DATA_DIR} created")

      label_idx = tsv_writer(df[[0,'label']].values.tolist(), label_file, idx=label_index, mode=mode)
      feature_idx = tsv_writer(df[[0,'feature']].values.tolist(), feature_file, idx=feature_index, mode=mode)
      return label_idx, feature_idx

    name = dataset_name.split('/')[-1].split('.')[0]

    hw_file = os.path.join(cfg.DATA_DIR, name+".hw.tsv")

    hw_df = pd.read_csv(hw_file, sep='\t', header=None, converters={1:ast.literal_eval}, index_col=0)
    
    if len(cfg.DATASETS.TEST) == 1:
      output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", os.path.splitext(model_name)[0])
    else:
      output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", name,
                                os.path.splitext(model_name)[0])

    if cfg.LIMIT_DATA==None or len(data_loader)<=cfg.LIMIT_DATA:
      pred_file = os.path.join(output_folder, 'predictions.tsv')
      df = pd.read_csv(pred_file, sep='\t',header = None, converters={1:json.loads})
      process(df)
    else:
      # pred_file = os.path.join(output_folder, 'predictions_1.tsv')
      # df = pd.read_csv(pred_file, sep='\t',header = None, converters={1:json.loads})
      lab_idx, feat_idx = 0, 0
      for idx in range(np.ceil(len(data_loader)/cfg.LIMIT_DATA).astype(int)):
        pred_file = os.path.join(output_folder, 'predictions_{}.tsv'.format(idx+1))
        df1 = pd.read_csv(pred_file, sep='\t',header = None, converters={1:json.loads})
        # df = pd.concat([df, df1], axis=0)
        lab_idx, feat_idx = process(df1, lab_idx, feat_idx, mode='a')

  for dataset_name, data_loader in zip(dataset_names, data_loaders_val):
    sub_get_feature(dataset_name, data_loader)

  print("Done")

#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#|     Format captions.txt: image,caption,split ngăn bởi (,)    |
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
def caption_txt2json(cfg):
  assert cfg.CAP_DIR.endswith(".txt")

  phase_list = ["train", "val", "test"]
  list_image_name = os.listdir(cfg.IMG_DIR)
  df_cap = pd.read_csv(cfg.CAP_DIR, sep=',')

  #Lọc caption ra theo image => Sử dụng cho tập Test 5k ảnh của Flickr30k
  # df_new = df_cap[df_cap['image']==list_image_name[0]]
  # for i in range(1, len(list_image_name)):
  #   df_new = pd.concat([df_new, df_cap[df_cap['image']==list_image_name[i]]], axis=0)
  
  def dump_caption(phase):
    df = df_cap[df_cap["split"]==phase]
    # Remove file extension '.jpg'
    df['imgID'] = df['image'].apply(lambda row : row[:-4])

    # Make clean captions
    df['caption'] = df['caption'].apply(lambda row : row.strip())

    # Create annotations
    annotations = [{"image_id" : df['imgID'].iloc[i], "id" : i, "caption" : df['caption'].iloc[i]} for i in range(df.shape[0])]
        
    # Create image annotations
    images = [{"id" : df['imgID'].iloc[i], "file_name" : df['image'].iloc[i]} for i in range(df.shape[0])]

    # Create coco_format
    coco_format = {"annotations" : annotations,
                  "images" : images,
                  "type": "captions",
                  "info": "dummy",
                  "licenses": "dummy"}

    # dump json
    with open(os.path.join(cfg.USED_DATA_DIR, phase+"_caption.json"), "w") as f:
      json.dump(annotations, f, separators=(', ', ': '), ensure_ascii = False)
      
    with open(os.path.join(cfg.USED_DATA_DIR, phase+"_caption_coco_format.json"), "w") as f:
      json.dump(coco_format, f, separators=(', ', ': '), ensure_ascii = False)
  
  for p in phase_list:
    dump_caption(p)

#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#|                  File caption_flickr8k.json                  |
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
def caption_json2json(cfg):
  assert cfg.CAP_DIR.endswith(".json")
  fp = open(cfg.CAP_DIR, "r")
  captions = json.load(fp)
  captions = captions["images"]

  def dump_caption(phase):
    annotations = []
    images = []
    for cap in captions:
      if cap["split"] == phase:
        imgID = cap["filename"][:-4]
        filename = cap["filename"]

        for c in cap["sentences"]:
          annot_dict = {}
          img_dict = {}

          annot_dict["image_id"] = imgID
          annot_dict["id"] = c["sentid"]
          annot_dict["caption"] = c["raw"]

          img_dict["id"] = imgID
          img_dict["file_name"] = filename

          annotations.append(annot_dict)
          images.append(img_dict)
    # Create coco_format
    coco_format = {"annotations" : annotations,
                  "images" : images,
                  "type": "captions",
                  "info": "dummy",
                  "licenses": "dummy"}

    # dump json
    with open(os.path.join(cfg.USED_DATA_DIR, phase+"_caption.json"), "w") as f:
      json.dump(annotations, f, separators=(', ', ': '), ensure_ascii = False)
      
    with open(os.path.join(cfg.USED_DATA_DIR, phase+"_caption_coco_format.json"), "w") as f:
      json.dump(coco_format, f, separators=(', ', ': '), ensure_ascii = False)
  
  phase_list = ["train", "val", "test"]
  for p in phase_list:
    dump_caption(p)

def dump_yaml_to_use(cfg):
  phase_list = ["train", "val", "test"]

  def sub_dump_yaml(phase):
    yaml_dict = {"label": phase+".label.tsv",
                "feature": phase+".feature.tsv",
                "caption": phase+"_caption.json"}

    with open(op.join(cfg.USED_DATA_DIR, phase+'.yaml'), 'w') as file:
        yaml.dump(yaml_dict, file)
  
  for p in phase_list:
    sub_dump_yaml(p)

def main():
    # cfg = get_config()
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend=cfg.DISTRIBUTED_BACKEND, init_method="env://"
        )
        synchronize()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    model_name = os.path.basename(ckpt)

    #Encode images and generate mini-tsv files
    build_mini_tsv(cfg)

    #Get predictions 
    run_test(cfg, model, args.distributed, model_name)

    #Create features and labels tsv
    get_feature_tsv(cfg, args.distributed, model_name)

    #Create caption from txt to json
    caption_json2json(cfg)

    #Create train.yaml
    dump_yaml_to_use(cfg)

if __name__ == "__main__":
    main()

