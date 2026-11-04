# Metrics

评测使用三层标签：

```text
basic → subordinate → fine_grained
```

## Accuracy

每个层级独立进行横向分类。模型对该层级全部类别产生 logits，预测类别与 batch 中对应的真实字符串标签相同则计为正确。

报告：

```text
accuracy.basic
accuracy.subordinate
accuracy.fine_grained
```

## Selectivity

对每张图片，取其真实 basic、subordinate、fine_grained 类别对应的三个 logits。对三者做比较，最高的层级计为模型选择的语义深度。

报告模型选择各层级的比例：

```text
selectivity.basic
selectivity.subordinate
selectivity.fine_grained
```

## Semantic depth

对真实路径上的三个 logits 做 softmax，并使用层级权重：

```text
basic = 0.0
subordinate = 0.5
fine_grained = 1.0
```

每张图片的 semantic depth 是三层概率加权和，最终报告全体样本的平均值。
