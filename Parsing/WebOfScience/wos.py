# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:23:50 2019

@author: steve
"""

import xlrd
import pickle

byDomains ={}
byAreas = {}

wb = xlrd.open_workbook("Data.xlsx")
sheet = wb.sheet_by_index(0)
first = True
for i in range(sheet.nrows):
    if first != True:
        byDomains[sheet.cell_value(i,6)] = sheet.cell_value(i, 3)
        byAreas[sheet.cell_value(i, 6)] = sheet.cell_value(i, 4)
        print(sheet.cell_value(i,3))
    first = False
    
print("length of domain dictionary: ", len(byDomains))
print("length of area dictionary: ", len(byAreas))

with open("webOfScienceDomains", "wb") as handle:
    pickle.dump(byDomains, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("webOfScienceAreas", "wb") as handle:
    pickle.dump(byAreas, handle, protocol=pickle.HIGHEST_PROTOCOL) 