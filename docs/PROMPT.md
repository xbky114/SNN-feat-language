# Prompt

`prompt.py` 提供唯一的 prompt 构造函数：

```python
from prompt import build_prompt

build_prompt(text, father=None)
```

- `text`：主类别名称，必须是非空字符串。
- `father`：可选父类别名称，必须是非空字符串或 `None`。

当 `father=None` 时，返回：

```text
a photo of {text}
```

当提供 `father` 时，返回：

```text
a photo of {text}, which is one kind of {father}
```
