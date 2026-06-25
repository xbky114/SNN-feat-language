# 1. Structure Overview

The file directory structure should be

```
--data
  --val (Imagenet1k dataset)
--models
  --model_name.pth
--conversion_frame_code
```

# 2. Running Example

base running mode is：

```
python main.py --config $path_to_config_file$
```

eg.

Evaluate ANN on ResNet34:

```
python main.py --config configs/resnet/resnet18_config_ann.yaml
```

Get threshold of SNN on ResNet34: Adjust the specific threshold mode for obtaining the threshold.

```
python main.py --config configs/resnet/resnet18_config_get_thre_channel.yaml
```

Test the converted SNN model of ResNet34:  Adjust the number of thresholds $n$in the specific configuration file. The threshold scaling parameter $c$(threshold_scale) defaults to 1.

```
python main.py --config configs/resnet/resnet18_config_test_channel_mth_diff.yaml
```

The same process applies to any model; you only need to replace the configuration file.