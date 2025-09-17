# Arrhenius_plot
- 在参数配置区修改基本参数如标题，字体字号等
- 将想要绘图的数据的文件名按照希望的顺序添加到files列表中
- files为空时会自动搜索当前文件夹中文件（数据结构一致时使用比较方便）  
- col_x和col_y是温度和电导率所在列，注意python中列从0开始计数。示例中温度在第1列，电导率在第2列
    >注意事项：温度单位为摄氏度，电导率单位为S/cm，活化能单位为eV，温度列和电导率列按照数据结构在arrhenius_plot.py文件中修改
- 修改label为自己期望的图例标签，这里经常会使用化合物名因此涉及格式，比如$\rm{CaCO_3}$,可以使用$\LaTeX$语法
- `use_fit=False`则绘制 $\sigma$ vs $1000/T$，`use_fit=True`则自动拟合所有数据并绘制 $\rm ln(\sigma T)$ vs $1000/T$,`show_Ea=True`会在图中显示结果（显示位置可能需要自己微调）
- 参数设置完毕后，将arrhenius_plot.py和数据文件放在同一文件夹，直接运行python文件即可
用法示例
```python
files = [
    {"filename": "1", "col_x": 0, "col_y": 1, "label": "undoped", "color": None, "marker": None, "linestyle": None,},#文件1，温度在第1列，电导率在第2列,图例为undoped
    {"filename": "2", "col_x": 0, "col_y": 1, "label": r"5% doped", "color": None, "marker": None, "linestyle": None,},#文件2，温度在第1列，电导率在第2列,图例为5% doped
    {"filename": "2", "col_x": 0, "col_y": 2, "label": r"10% doped", "color": None, "marker": None, "linestyle": None,},#文件2，温度在第1列，电导率在第3列,图例为10% doped
]
```

