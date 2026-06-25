import torch
import torchvision.models.detection as detection_models
import inspect
import os
model_name_list = ['fcos_resnet50_fpn','retinanet_resnet50_fpn','retinanet_resnet50_fpn_v2']
save_name_list = ['fcos_resnet50_fpn','retinanet_resnet50_fpn','retinanet_resnet50_fpn_v2']

if __name__ == '__main__':
    # 获取所有目标检测模型的名称
    model_functions = inspect.getmembers(detection_models, inspect.isfunction)
    detection_model_names = [name for name, func in model_functions]

    print("可用的目标检测模型名称：")
    for idx, model_name in enumerate(detection_model_names, 1):
        print(f"{idx}. {model_name}")

    for i in range(0, len(model_name_list)):
        print(f"\n正在下载并保存模型：{model_name_list[i]}")
        # 加载预训练模型
        model = getattr(detection_models, model_name_list[i])(pretrained=True)
        model.eval()

        # 定义保存路径
        save_dir = '../models/det/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{save_name_list[i]}.pth")

        # 保存模型的状态字典
        torch.save({'model': model.state_dict()}, save_path)
        print(f"模型已保存到 {save_path}")

        # 验证加载模型
        loaded_model = getattr(detection_models, model_name_list[i])(pretrained=False)
        loaded_model.load_state_dict(torch.load(save_path)["model"])
        loaded_model.eval()
        print(f"模型 {model_name_list[i]} 已成功加载并验证。")
