# Install
```
pip install -U scikit-learn pandas matplotlib seaborn
python lr.py
python unsupervised.py
...
```

# Linear Regression
```
RMSE: 0.13388594148552893
R²: 0.00037485248096846835
```

# PCA
```
Principal Component 1: 0.27
Principal Component 2: 0.20
Principal Component 3: 0.20
Principal Component 4: 0.18
Principal Component 5: 0.15

|    PC1   |    PC2   |    PC3   |    PC4   |    PC5   | Accepted | Rejected | Delivery_Date |
|----------|----------|----------|----------|----------|----------|----------|---------------|
| 1.945767 | 0.205579 | 0.118121 | 0.046944 | 1.366391 |    NaN   |    NaN   |      NaN      |
| 2.993892 | 2.718035 | 1.859423 | 1.619489 | -0.881580|    NaN   |    NaN   |      NaN      |
| 0.920100 | 1.101846 | -0.856055| 0.231532 | 0.756976 |    NaN   |    NaN   |      NaN      |
| 0.328679 | 0.192849 | 2.240902 | -1.410089| -0.625093|    NaN   |    NaN   |      NaN      |
| 0.001775 | -0.024006| 0.317994 | 0.519875 | -0.483578|    NaN   |    NaN   |      NaN      |

```

# Random Forest 
```
RMSE: 0.1393007362810676
R²: -0.08211655978299626
```


# Unsupervised
```
| Cluster | Accepted  | Rejected  | Avg_Temp(C) | Precipitation(ml) | Wind_Speed(km/hr) |     lat    |     lng    |
|---------|-----------|-----------|-------------|--------------------|-------------------|------------|------------|
|    0    |  0.017833 |  0.982167 |   5.950286  |       1.436167     |      18.143022    | 49.637209  |  6.092630  |
|    1    |  0.018613 |  0.981387 |  17.691048  |       0.815578     |      13.591073    | 49.661753  |  6.097524  |
|    2    |  0.017513 |  0.982487 |   9.138933  |      14.444952     |      19.709084    | 49.657272  |  6.097096  |

```

# Supervised (Also RF)
```
RMSE: 0.1392886684539173
R²: -0.08192907721402731
```