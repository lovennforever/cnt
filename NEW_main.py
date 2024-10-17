# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:05:48 2024

@author: Administrator
"""

import new_h

if __name__== "__main__":
    base_data_path = r"./data/avg_se"

# =============================================================================
#     AVG_SE.CntMinValue(base_data_path)
#     AVG_SE.C60DataChange(base_data_path)
#     AVG_SE.CurveC60DensityRelativeCoordinates(base_data_path)
#     AVG_SE.CalAvgSe(base_data_path)
#     AVG_SE.CurveAvgSe(base_data_path)
# =============================================================================


    DealWithTensionFile(base_data_path) 
    CntDensityData(base_data_path) 
    C60DensityData(base_data_path)
    C60DensityDataRelativeCoordinates(base_data_path)
    ProcessDensityFiles(base_data_path)
    MergeC60CntMinXdata(base_data_path)
    HistogramC60Cnt(base_data_path)