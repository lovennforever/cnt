# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:27:57 2024

@author: Administrator
"""
#import AVG_SE
#import h
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

#     h.DealWithTensionFile(base_data_path)     
#     h.CntDensityData(base_data_path)
#     h.C60DensityData(base_data_path)
#     h.C60DensityDataRelativeCoordinates(base_data_path)
#     h.ExtractMinTimeC60(base_data_path)
#     h.ExtractFiledensity_min_values_c60_corresponding_xMinTimeC60(base_data_path)
#     h.MergeC60CntMinXdata(base_data_path)   
#     h.HistogramC60Cnt(base_data_path)
# =============================================================================
        
    new_h.DealWithTensionFile(base_data_path) 
    new_h.CntDensityData(base_data_path) 
    new_h.C60DensityData(base_data_path)
    new_h.C60DensityDataRelativeCoordinates(base_data_path)
    new_h.ProcessDensityFiles(base_data_path)
    new_h.MergeC60CntMinXdata(base_data_path)
    new_h.HistogramC60Cnt(base_data_path)