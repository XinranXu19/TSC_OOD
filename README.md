# TSC_OOD

### 训练模型

```python
python train.py --data EEG --scrID 2
```

### OOD测试

当method为source时，直接将模型迁移到target域，不进行tent更新；当method为tent时，利用tent方法在target域进行更新。

```python
python mytent.py --data EEG
				 --Sdata EEG
    			 --scrID 1
        		 --trgID 11
            	 --steps 5
                 --method tent
```

以上为EEG数据集 1-> 11 运行示例 (使用tent)。

```python
python mytent.py --data FordA
				 --Sdata FordB
                 --method source
```

以上为数据集 FordB-> FordA 运行示例（不使用tent）。

