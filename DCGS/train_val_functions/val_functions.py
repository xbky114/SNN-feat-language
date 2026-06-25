import torch
from utils import reset
from statics import SOPMonitor
from tqdm import tqdm

def val_ann_classfication(model, test_loader, device, args=None):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((test_loader)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            print(batch_idx,100 * correct / total)
        final_acc = 100 * correct / total
    return final_acc

def val_snn_classfication(model, test_loader, device, args=None):
    correct = 0
    total = 0
    model.eval()
    all_correct = [0 for i in range(model.T)]
    all_total = [0 for i in range(model.T)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((test_loader)):
            reset(model)
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # 准确率
            for i in range(model.T):
                outputs_T = outputs[:i+1].mean(0)
                _, predicted = outputs_T.cpu().max(1)
                all_correct[i] += float(predicted.eq(targets).sum().item())
                all_total[i] += float(targets.size(0))
            
            print('批次：' , batch_idx, ' 最终准确率：', 100 * all_correct[-1] / all_total[-1])
            print('平均准确率: ' + ', '.join([str(100 * all_correct[i] / all_total[i]) for i in range(model.T)]))
            
        final_acc = 100 * all_correct[-1] / all_total[-1]
    return final_acc


def val_snn_classfication_with_sop(model, test_loader, device, args=None):
    correct = 0
    total = 0
    mon = SOPMonitor(model,step_mode=model.step_mode,T=model.T,coding_type=model.coding_type)
    model.eval()
    all_sops = [0 for i in range(model.T)]
    all_tots = [0 for i in range(model.T)]
    all_correct = [0 for i in range(model.T)]
    all_total = [0 for i in range(model.T)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="推理")):
            reset(model)
            inputs = inputs.to(device)
            outputs = model(inputs)
            # 准确率
            for i in range(model.T):
                outputs_T = outputs[:i+1].mean(0)
                _, predicted = outputs_T.cpu().max(1)
                all_correct[i] += float(predicted.eq(targets).sum().item())
                all_total[i] += float(targets.size(0))
            
            print('批次：' , batch_idx, ' 最终准确率：', 100 * all_correct[-1] / all_total[-1])
            print('平均准确率: ' + ', '.join([str(100 * all_correct[i] / all_total[i]) for i in range(model.T)]))
            
            # 计算能耗
            now_sop = [0 for i in range(model.T)]
            now_tot = [0 for i in range(model.T)]
            for name in mon.monitored_layers:
                sublist = mon[name]
                if 'diff' in model.coding_type:
                    sublist = sublist[1:]
                # print(name,len(sublist))
                if len(sublist)>0:
                    for t in range(model.T):
                        now_sop[t]+=sublist[t][0]
                        now_tot[t]+=sublist[t][1]
                        # print(t,sublist[t][0],sublist[t][1],round(float(sublist[t][0])/float(sublist[t][1]),4))
            fire_list = []
            energy_list = []
            for i in range(model.T):
                all_sops[i]+=now_sop[i]
                all_tots[i]+=now_tot[i]
            for i in range(model.T):
                tmp = float(sum(all_sops[:i+1])/sum(all_tots[:i+1]))
                fire_list.append(tmp)
                energy_list.append((i+1)*0.9/4.6*tmp)
            print('瞬时发射率: ' + ', '.join([str(round(float(all_sops[i]/all_tots[i]),4)) for i in range(model.T)]))
            print('平均发射率: ' + ', '.join([str(round(i,4)) for i in fire_list]))
            print('总能耗: ' + ', '.join([str(round(i,4)) for i in energy_list]))
            mon.clear_recorded_data()
            
            
        final_acc = 100 * all_correct[-1] / all_total[-1]
    return final_acc


def val_ann_object_detection(model, test_loader, device, args=None, score_threshold=0.05):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_results = []
    for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="推理")):
        if batch_idx>10:
            break
        images = list(image.to(device) for image in images)
        # targets 是一个元组，每个元素是一个列表
        if len(targets[0]) == 0:
            # 如果没有目标（即没有标注），跳过
            continue
        image_id = targets[0][0]['image_id']
        
        with torch.no_grad():
            outputs = model(images)
        
        for output in outputs:
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            # 过滤低置信度的预测
            keep = scores >= score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                coco_results.append({
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': [float(x_min), float(y_min), float(width), float(height)],
                    'score': float(score)
                })
    print(f"推理完成，收集到 {len(coco_results)} 个预测结果。")
    # 保存预测结果
    import os
    import json
    coco_annotation_file = os.path.join(args.dataset_path, 'COCO/annotations/instances_val2017.json')
    prediction_file = args.save_name+'.json'
    with open(prediction_file, 'w') as f:
        json.dump(coco_results, f)
    print(f"预测结果已保存到 {prediction_file}")
    coco = COCO(coco_annotation_file)
    coco_pred = coco.loadRes(prediction_file)
    coco_eval = COCOeval(coco, coco_pred, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_results

def val_snn_object_detection(model, test_loader, device, args=None, score_threshold=0.05):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_results = [[] for i in range(args.time)]
    for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="推理")):
        if batch_idx>10:
            break
        images = list(image.to(device) for image in images)
        # targets 是一个元组，每个元素是一个列表
        if len(targets[0]) == 0:
            # 如果没有目标（即没有标注），跳过
            continue
        image_id = targets[0][0]['image_id']
        
        with torch.no_grad():
            outputs_all = model(images)
        for i in range(args.time):
            outputs = outputs_all[i]
            for output in outputs:
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                # print(boxes.shape,scores.shape,labels.shape)
                # print(boxes,scores,labels)

                # 过滤低置信度的预测
                keep = scores >= score_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    coco_results[i].append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [float(x_min), float(y_min), float(width), float(height)],
                        'score': float(score)
                    })
    
    # 保存预测结果
    import os
    import json
    for i in range(args.time):
        print(f"推理完成，前{i}步收集到 {len(coco_results[i])} 个预测结果。")
        coco_annotation_file = os.path.join(args.dataset_path, 'COCO/annotations/instances_val2017.json')
        prediction_file = args.save_name+'.json'
        with open(prediction_file, 'w') as f:
            json.dump(coco_results[i], f)
        coco = COCO(coco_annotation_file)
        coco_pred = coco.loadRes(prediction_file)
        coco_eval = COCOeval(coco, coco_pred, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_results


def _get_clip_eval_paths(args):
    clip_image_path = getattr(args, 'clip_image_path', '../CLIP/CLIP.png')
    clip_checkpoint = getattr(args, 'clip_checkpoint', None)
    labels = getattr(args, 'clip_labels', ["a diagram", "a dog", "a cat"])
    return clip_image_path, clip_checkpoint, labels


def val_ann_clip(model, test_loader, device, args=None):
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            print(f"batch {batch_idx}: feature shape={tuple(outputs.shape)}, norm={outputs.norm(dim=-1).mean().item():.4f}")
            break
    return 0.0


def val_snn_clip(model, test_loader, device, args=None):
    import os
    import sys

    import clip
    import numpy as np
    from PIL import Image

    clip_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'CLIP'))
    if clip_root not in sys.path:
        sys.path.insert(0, clip_root)

    from models.clip_wrapper import CLIPWithSNNVisual, backbone_cosine, encode_ann_backbone
    from utils import reset

    clip_image_path, clip_checkpoint, labels = _get_clip_eval_paths(args)
    clip_checkpoint = clip_checkpoint or getattr(args, 'load_name', None)

    ann_model, preprocess = clip.load("RN50", device=device, jit=False, download_root=os.path.expanduser('~/.cache/clip'))
    if clip_checkpoint and os.path.exists(clip_checkpoint) and not clip_checkpoint.endswith('.pth'):
        ann_model, _ = clip.load(clip_checkpoint, device=device, jit=False)
    ann_model = ann_model.float()

    wrapper = CLIPWithSNNVisual(model, ann_model).to(device).eval()

    image = preprocess(Image.open(clip_image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        probs_ann = ann_model(image, text)[0].softmax(dim=-1).cpu().numpy()
        reset(model)
        probs_snn = wrapper(image, text)[0].softmax(dim=-1).cpu().numpy()

        feat_ann = ann_model.encode_image(image)
        reset(model)
        feat_snn = wrapper.encode_image(image)
        cosine = torch.nn.functional.cosine_similarity(feat_ann, feat_snn).item()

        backbone_ann = encode_ann_backbone(ann_model.visual, image)
        reset(model)
        backbone_snn = wrapper.encode_backbone(image)
        cosine_backbone = backbone_cosine(backbone_ann, backbone_snn)

    print("Labels:", labels)
    print("ANN probs:", probs_ann)
    print("SNN probs:", probs_snn)
    print("Abs diff:", np.abs(probs_ann - probs_snn))
    print("Cosine(image_features):", cosine)
    print("Cosine(backbone_features):", cosine_backbone)
    print("ANN top-1:", labels[int(probs_ann.argmax())])
    print("SNN top-1:", labels[int(probs_snn.argmax())])
    return cosine