# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:23:50 2019

@author: steve
"""

import xlrd
import pickle

byDomains ={}
byAreas = {}

wb = xlrd.open_workbook("../../Data/WOS.xlsx")
sheet = wb.sheet_by_index(0)
first = True
for i in range(sheet.nrows):
    if first != True:
        byDomains[sheet.cell_value(i,6)] = sheet.cell_value(i, 3)
        byAreas[sheet.cell_value(i, 6)] = sheet.cell_value(i, 4)
        # print(sheet.cell_value(i,3))
    first = False
    
# print("length of domain dictionary: ", len(byDomains))
# print("length of area dictionary: ", len(byAreas))

mainData = byDomains

mainDict = {}
for i in mainData:
    key = mainDict.get(i)
    if key == None:
        mainDict[i] = [mainData[i][:-1]]
    else:
        mainDict[i].append(mainDict[i][:-1])

    # print(mainData[i])
    # print("\n------------------------------------\n")

mainList = []
for i in mainDict:
    dataSet = [i, mainDict[i]]
    mainList.append(dataSet)

# for i in mainList:
# 	print(i)
# 	print("\n------------------------------------\n")
pickle.dump(mainList, open("../../Data/WOS.p", "wb" ))