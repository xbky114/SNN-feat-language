import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn


def seed_all(seed=2024):
    """
    Set a fixed seed for reproducibility across random, numpy, and torch modules.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def get_logger(logger, logger_path, verbosity=1, name=None):
    """
    Initializes and returns a logger object based on the specified parameters.

    Args:
        logger (bool): Flag to enable or disable logging. If False, returns None.
        logger_path (str): Path where the log file will be created. Directories are created if they don't exist.
        verbosity (int): Logging verbosity level (0=DEBUG, 1=INFO, 2=WARNING). Default is 1.
        name (str): Optional name for the logger. Default is None.

    Returns:
        logging.Logger: Configured logger instance if logging is enabled, or None if disabled.

    Example:
        # Enable logging and set up a logger
        logger = get_logger(
            logger=True,
            logger_path="logs/app.log",
            verbosity=1,
            name="AppLogger"
        )
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")

        # Disable logging
        logger = get_logger(logger=False, logger_path="logs/app.log")
        if logger is None:
            print("Logger is disabled.")
    """
    if logger:
        # Ensure the directory for the logger path exists
        log_dir = os.path.dirname(logger_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Create the directory if it doesn't exist

        # Define log level mapping
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}

        # Create a formatter for log messages
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )

        # Create and configure the logger
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        # File handler for writing logs to a file
        file_handler = logging.FileHandler(logger_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler for writing logs to the console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Log initialization messages
        logger.info("Logger initialized")
        logger.info(f"Log file path: {logger_path}")
    else:
        logger = None
        print("Logging is disabled")

    return logger


class MergeTemporalDim(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)
    
    
class ExpandTemporalDim_dict(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
    
    def forward(self, x_dict: dict):
        # 创建一个新的字典来保存转换后的输出
        y_dict = {}
        # 遍历输入字典中的每一个键值对
        for key, x_seq in x_dict.items():
            if not isinstance(x_seq, torch.Tensor):
                raise ValueError(f"Expected a torch.Tensor but got {type(x_seq)} for key '{key}'")
            # 计算新的形状，并进行验证
            y_shape = [self.T, int(x_seq.shape[0] / self.T)]
            y_shape.extend(x_seq.shape[1:])
            y_dict[key] = x_seq.view(y_shape)
        return y_dict
def reset(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = reset(module)
        if hasattr(module, "reset"):
            model._modules[name].reset()
    return model

class MyAt(nn.Module):
    def __init__(self):
        super(MyAt, self).__init__()
    def forward(self, x, y):
        return x @ y

class MyMul(nn.Module):
    def __init__(self):
        super(MyMul, self).__init__()
    def forward(self, x, y):
        return x * y
    
    
class MyatSequential(nn.Module):
    def __init__(self, neuron1, neuron2, module):
        super().__init__()
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        self.module = module
    def forward(self, *inputs):
        return self.module(self.neuron1(inputs[0]),self.neuron2(inputs[1]))

# class MySequential(nn.Module):
#     def __init__(self, *modules):
#         """
#         *modules: 依次传入的子模块，可以是任意 nn.Module
#         """
#         super().__init__()
#         for idx, module in enumerate(modules):
#             self.add_module(str(idx), module)

#     def forward(self, *inputs):
#         """
#         *inputs: 可变参数，多输入
#         """
#         # 将所有子模块按照顺序执行
#         for module in self._modules.values():
#             # 使用当前 inputs 调用下一个子模块
#             #   - 如果 inputs 是个 tuple，会被展开成多个参数
#             #   - 如果 inputs 是单个张量，需要先把它包装成 tuple，才能用 *tuple 解包
#             outputs = module(*inputs)  
            
#             # 将输出再次包装成 tuple，以保证后续链式调用的一致性
#             if not isinstance(outputs, tuple):
#                 outputs = (outputs,)
                
#             # 下一轮循环就以本次的 outputs 作为 inputs
#             inputs = outputs
        
#         # 最后，如果只包含一个元素，就直接返回那个张量；否则返回 tuple
#         if len(inputs) == 1:
#             return inputs[0]
#         else:
#             return inputs

def get_modules(nowname,model):
    flag = 0
    for name, module in model._modules.items():
        if flag==0:
            print(model.__class__.__name__.lower(),end=' ')
            flag=1
        print(module.__class__.__name__.lower(),end=' ')
    if flag==1:
        print('')
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = get_modules(name,module)
    return model