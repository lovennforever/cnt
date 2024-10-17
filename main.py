# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:27:57 2024

@author: Administrator
"""
import AVG_SE

if __name__== "__main__":
    base_data_path = r"./data/avg_se"

    AVG_SE.CntMinValue(base_data_path)
    AVG_SE.C60DataChange(base_data_path)
    AVG_SE.CurveC60DensityRelativeCoordinates(base_data_path)
    AVG_SE.CalAvgSe(base_data_path)
    AVG_SE.CurveAvgSe(base_data_path)