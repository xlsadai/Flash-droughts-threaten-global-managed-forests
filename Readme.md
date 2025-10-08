# Flash droughts threaten global managed forests
Jianzhuang Pang <sup>1, 2, 3</sup>, Hang Xu <sup>1, 2, 3, *</sup>, Yang Xu <sup>1, 2, 3</sup>, 
Yifan Zhang <sup>1, 2, 3</sup>, Xiaoyun Wu <sup>1, 2, 3</sup>, Kexin Li <sup>1, 2, 3</sup>, 
Zhiqiang Zhang <sup>1, 2, 3, *</sup>

<sup>1</sup> Jixian National Forest Ecosystem Observation and Research Station, CNERN, Beijing Forestry University, 
Beijing 100083, P.R. China  
<sup>2</sup> Key Laboratory of Soil and Water Conservation and Desertiﬁcation Combating, State Forestry and Grassland 
Administration, Beijing 100083, P.R. China  
<sup>3</sup> School of Soil and Water Conservation, Beijing Forestry University, Beijing 100083, P.R. China

**\*Correspondence**  
Dr. Hang Xu, School of Soil and Water Conservation, Beijing Forestry University, Beijing, China. 
Email: hangxu@bjfu.edu.cn  
Dr. Zhiqiang Zhang, School of Soil and Water Conservation, Beijing Forestry University, Beijing, China. 
Tel: 0086-10-62338828 Email: zhqzhang@bjfu.edu.cn 


## Description
The code for plotting and data analysis used in the manuscript.
- **Developed by:** [Jianzhuang Pang](xlsadaii@bjfu.edu.cn)
- **Language(s) (Programming):** [Python](https://www.python.org/)
- **Language(s) (NLP):** English
- **License:** [MIT License](https://mit-license.org/)

## How to Get Start
The results of the paper can be reproduced by using the following code in **_main.py_**.    
Before using the code, you should [download the data](https://figshare.com/s/00a907a42f869aa18d74) for analysis and 
replace **_manuscript_data_** with the folder path where the data is stored, and set **_figure_path_** to the folder 
path where you wish to save the results.
- Fig.1. Spatial patterns and latitudinal variations in flash drought characteristics across global forests.
```python
  figure_1abcd()  # Subplot a, b, c, and d
  figure_1e()  # Subplot e
```
- Fig. 2. Impacts of flash drought on global forest.
```python
  figure_2a()  # Subplot a
  figure_2b()  # Subplot b
  figure_2c()  # Subplot c
```
- Fig. 3 Interactive effects of regulating factors on forest responses to flash drought between intact and 
managed forests.
```python
  figure_3()
```
- Supplementary Figure 1. Spatial distribution of temporal trends in flash drought characteristics.
```python
  supplementary_figure_1_ace()  # Subplot a, c, and e
  supplementary_figure_1_bdf()  # Subplot b, d, and f
  supplementary_figure_1_g()  # Subplot g
```
- Supplementary Figure 2. Spatial distribution of (a) normalized temperature anomalies, (b) normalized precipitation 
anomalies, and (c) consistency between precipitation and temperature anomalies during flash droughts.
```python
  supplementary_figure_2_ab()  # Subplot a and b
  supplementary_figure_2_c()  # Subplot c
```
- Supplementary Figure 3. Spatial patterns and latitudinal variations in flash drought characteristics across global 
forests (soil moisture-based).
```python
  supplementary_figure_3abcd()  # Subplot a, b, c, and d
  supplementary_figure_3_e()  # Subplot e
```
- Supplementary Figure 4. The spatial distribution of flash drought frequency.
```python
  supplementary_figure_4()
```
- Supplementary Figure 5. Spatial patterns of mean anomalies during flash droughts. 
```python
  supplementary_figure_5_abc()  # Subplot a, b, and c
  supplementary_figure_5_de()  # Subplot d and e
```
- Supplementary Figure 6. Impacts of flash droughts on global forests (soil moisture-based).
```python
  supplementary_figure_6_a() # Subplot a
  supplementary_figure_6_b() # Subplot b
  supplementary_figure_6_c() # Subplot c
```
- Supplementary Figure 7. Normalized Difference Vegetation Index anomalies (ΔNDVI) during (a) Standardized 
Precipitation-Evapotranspiration Index-base, (b) soil moisture-base flash droughts relative to concurrent normal 
conditions under different forest types.
```python
  supplementary_figure_7_a() # Subplot a
  supplementary_figure_7_b() # Subplot b
```
- Supplementary Figure 8. The individual effects of regulating factors on forest response (ΔNDVI) to flash droughts.
```python
  supplementary_figure_8()
```
- Supplementary Figure 9. Normalized Difference Vegetation Index anomalies (ΔNDVI) during flash droughts relative 
to concurrent normal conditions under different (a) Forest management; (b) Forest management practices
```python
  supplementary_figure_9_a()
  supplementary_figure_9_b()
```
- Supplementary Figure 12. Performance evaluation of the extreme gradient boosting model.
```python
  supplementary_figure_12()
```

## Technical Specifications
#### Hardware
- CPU: Intel Core i9-13900
- GPU: NVIDIA GeForce RTX 4090
- RAM: 128 GB
- Storage: 10 TB

#### Software
- IDE: PyCharm Community Edition 2023.3.5
- Operating System: Windows 11
- Python Version: Python 3.12.2
- Libraries/Frameworks:
  - pandas=2.2.1
  - numpy=1.26.4
  - scipy=1.12.0
  - statsmodels=0.14.0
  - joblib=1.2.0
  - seaborn=0.12.2
  - matplotlib=3.8.4
  - xarray=2025.1.1
  - rasterio=1.3.9
  - cartopy=0.22.0
  - dask=2023.11.0
  - xgboost=2.0.3
  - scikit-learn=1.3.0

## Citation
Pang JZ, Xu H, Xu Y, Zhang YF, Wu XY, Li KX, and Zhang ZZ. 2025. Flash droughts threaten global managed forests. 
Nature Communications.