--dataloader.py # default imagenet dataloader and calibration dataloader.
receive path: a img dir like /data/home/bky/SNN-feat-language/data/imagenet/val

--CLIPWithSNNVisual.py: class CLIPWithSNNVisual. a wrapper class
    input: ann_clip, snnartifact
    def encode_image(self, img, T):
        reset snn
        iter T # T can differ from T in conversion
        feats[t] = snn(t)
        return feats
    def forward(self, img_feat, text_feat):
        -> origin CLIP logic
    def calc_logits_through_time(self, img, texts, T):
        img_feats = selfencode_image(img, T)
        text_feat = self.encode_texts
        iter T
        img_feat = img_feats[t]
        ogits = self(img_feat, text_feat)
        all_logits[t] = logits
        return all_logits

--get_snn.py, ignore by git   

--wordnet_label.py
    build the horizontal and vertical label

--eval_ann.py: classification acc

--eval_snn.py

metrics/
