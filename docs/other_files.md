--dataloader.py # default dataloader and calibration dataloader.

--CLIPWithSNNVisual.py: class CLIPWithSNNVisual. a wrapper class
    input: ann_clip, snnartifact, T
    forward:
        -> reset snn
        -> iter T
        -> all_logits[t] = CLIP logit using snn(x,t) as visual feature
        -> return all_logits

--get_snn.py, ignore by git   

--wordnet_label.py
    build the horizontal and vertical label

--eval.py

metrics/
