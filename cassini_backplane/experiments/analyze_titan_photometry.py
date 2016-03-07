import colorsys
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sciopt
import scipy.interpolate as interp

import imgdisp

import oops
import oops.inst.cassini.iss as iss

from cb_gui_offset_data import *
from cb_offset import *
from cb_util_file import *


IMAGE_LIST_CLEAR = [
    ### CLEAR FILTERS
    'COISS_2058/data/1634043408_1634144961/W1634064022_1_CALIB.IMG', #  10.59  256  10.59_W1634064022_1
    'COISS_2033/data/1561837467_1561882811/W1561882645_1_CALIB.IMG', #  11.72 1024  11.72_W1561882645_1
    ('COISS_2032/data/1560468671_1560490232/W1560469688_1_CALIB.IMG',#  16.18  256  16.18_W1560469688_1
     'COISS_2032/data/1560468671_1560490232/W1560471337_1_CALIB.IMG'),
    'COISS_2009/data/1487070070_1487120067/W1487070087_8_CALIB.IMG', #  19.94 1024  19.94_W1487070087_8
    ('COISS_2032/data/1559081809_1559100520/W1559090633_1_CALIB.IMG',#  24.52 1024  24.52_W1559090633_1
     'COISS_2032/data/1559081809_1559100520/W1559092021_1_CALIB.IMG'),
    'COISS_2056/data/1625791792_1625815191/W1625798841_1_CALIB.IMG', #  27.99  256  27.99_W1625798841_1
    'COISS_2032/data/1557721167_1557737986/W1557729551_1_CALIB.IMG', #  28.65 1024  28.65_W1557729551_1
    'COISS_2019/data/1515806369_1515985762/W1515969272_1_CALIB.IMG', #  35.18 1024  35.18_W1515969272_1
    'COISS_2031/data/1556223784_1556359484/W1556342475_1_CALIB.IMG', #  37.17 1024  37.17_W1556342475_1
    'COISS_2074/data/1712333148_1713050840/N1712583648_1_CALIB.IMG', #  38.93 1024 NAV +350  38.93_N1712583648_1
    'COISS_2055/data/1622711732_1623166344/W1623046679_1_CALIB.IMG', #  44.44  256  44.44_W1623046679_1
    'COISS_2031/data/1554952789_1554965182/W1554961095_1_CALIB.IMG', #  45.48  512  45.48_W1554961095_1
    'COISS_2068/data/1683616251_1683727987/W1683640370_1_CALIB.IMG', #  47.71 1024  47.71_W1683640370_1
    'COISS_2030/data/1553532865_1553602892/W1553595036_1_CALIB.IMG', #  52.71  256  52.71_W1553595036_1
    'COISS_2010/data/1490886737_1490969103/W1490952354_2_CALIB.IMG', #  57.33 1024  57.33_W1490952354_2
    'COISS_2024/data/1530431891_1530501147/W1530483314_1_CALIB.IMG', #  60.94 1024  60.94_W1530483314_1
    'COISS_2026/data/1539140799_1539203530/W1539146659_1_CALIB.IMG', #  64.59 1024  64.59_W1539146659_1
    'COISS_2074/data/1713139654_1713235584/N1713218752_1_CALIB.IMG', #  69.07 1024 NAV +350  69.07_N1713218752_1
    'COISS_2026/data/1540495579_1540576032/W1540513147_1_CALIB.IMG', #  71.47 1024  71.47_W1540513147_1
    'COISS_2029/data/1548774401_1548803100/W1548800888_1_CALIB.IMG', #  76.41 1024  76.41_W1548800888_1
    'COISS_2049/data/1604264777_1604402429/W1604385250_1_CALIB.IMG', #  79.89 1024  79.89_W1604385250_1
    'COISS_2045/data/1589301026_1589445623/W1589361573_1_CALIB.IMG', #  81.62 1024  81.62_W1589361573_1
    'COISS_2058/data/1636752814_1636890977/N1636847461_1_CALIB.IMG', #  87.30 1024 NAV +350  87.30_N1636847461_1
    'COISS_2023/data/1526797460_1526852621/W1526797524_1_CALIB.IMG', #  91.76 1024  91.76_W1526797524_1
    'COISS_2059/data/1637959435_1638085617/N1638002556_1_CALIB.IMG', #  96.25 1024 NAV +350  96.25_N1638002556_1
    'COISS_2050/data/1605837398_1606077468/W1605841151_1_CALIB.IMG', # 100.03  256 100.03_W1605841151_1
    'COISS_2028/data/1548522106_1548756214/W1548687870_1_CALIB.IMG', # 105.00 1024 105.00_W1548687870_1
    'COISS_2071/data/1701985344_1702247372/N1702239215_1_CALIB.IMG', # 107.05 1024 NAV +400 107.05_N1702239215_1
    'COISS_2058/data/1635278317_1635374244/N1635365802_1_CALIB.IMG', # 109.77 1024 NAV +420 109.77_N1635365802_1
    'COISS_2024/data/1530502947_1530762570/W1530568094_1_CALIB.IMG', # 114.55 1024 114.54_W1530568094_1
    'COISS_2026/data/1538857861_1539059851/W1539033793_1_CALIB.IMG', # 117.36 1024 117.36_W1539033793_1
    'COISS_2034/data/1563027386_1563490622/W1563472949_1_CALIB.IMG', # 123.50 1024 123.50_W1563472949_1
    'COISS_2031/data/1554840347_1554952668/W1554888844_1_CALIB.IMG', # 129.76  256 129.76_W1554888844_1
    'COISS_2039/data/1575313792_1575487821/W1575463985_1_CALIB.IMG', # 137.66  256 137.66_W1575463985_1
    'COISS_2086/data/1761119366_1762878019/N1762149932_1_CALIB.IMG', # 154.44 1024 NAV +350 154.44_N1762149932_1
    'COISS_2023/data/1527484252_1528428866/N1527944332_1_CALIB.IMG', # 163.09 1024 NAV +450 - OVEREXPOSED? 163.09_N1527944332_1
]

IMAGE_LIST_BY_PHASE = [
    # 12
    'COISS_2033/data/1561837467_1561882811/W1561880145_1_CALIB.IMG', # VIO
    
    # 13
    'COISS_2057/data/1629783492_1630072138/W1629929416_1_CALIB.IMG', # VIO
    'COISS_2057/data/1629783492_1630072138/W1629929449_1_CALIB.IMG', # BL1
    'COISS_2057/data/1629783492_1630072138/W1629929482_1_CALIB.IMG', # GRN
    'COISS_2057/data/1629783492_1630072138/W1629929515_1_CALIB.IMG', # RED
    'COISS_2007/data/1477438740_1477481438/W1477471954_6_CALIB.IMG', # IR3
    'COISS_2057/data/1629783492_1630072138/W1629930385_1_CALIB.IMG', # CB2
    'COISS_2057/data/1629783492_1630072138/W1629929871_1_CALIB.IMG', # CB3
    'COISS_2057/data/1629783492_1630072138/W1629930219_1_CALIB.IMG', # MT2
    'COISS_2057/data/1629783492_1630072138/W1629929965_1_CALIB.IMG', # MT3
    
    # 15
    'COISS_2032/data/1560496833_1560553326/W1560496994_1_CALIB.IMG', # VIO
    
    # 16
    'COISS_2068/data/1680805782_1681997642/W1681926856_1_CALIB.IMG', # VIO
    'COISS_2068/data/1680805782_1681997642/W1681936403_1_CALIB.IMG', # VIO
    'COISS_2068/data/1680805782_1681997642/W1681945950_1_CALIB.IMG', # VIO
    'COISS_2057/data/1629783492_1630072138/W1629929515_1_CALIB.IMG', # RED
    'COISS_2068/data/1680805782_1681997642/W1681910412_1_CALIB.IMG', # RED
    'COISS_2068/data/1680805782_1681997642/W1681916843_1_CALIB.IMG', # RED
    'COISS_2068/data/1680805782_1681997642/W1681926406_1_CALIB.IMG', # RED
    'COISS_2068/data/1680805782_1681997642/W1681935953_1_CALIB.IMG', # RED
    'COISS_2068/data/1680805782_1681997642/W1681945500_1_CALIB.IMG', # RED
    
    # 18               
    'COISS_2068/data/1680805782_1681997642/W1681908315_1_CALIB.IMG', # VIO
    'COISS_2068/data/1680805782_1681997642/W1681910478_1_CALIB.IMG', # BL1
    'COISS_2068/data/1680805782_1681997642/W1681908381_1_CALIB.IMG', # GRN
    'COISS_2068/data/1680805782_1681997642/W1681908414_1_CALIB.IMG', # RED
    'COISS_2068/data/1680805782_1681997642/W1681909321_1_CALIB.IMG', # CB2
    'COISS_2068/data/1680805782_1681997642/W1681909505_1_CALIB.IMG', # CB3
    'COISS_2068/data/1680805782_1681997642/W1681909361_1_CALIB.IMG', # MT2
    'COISS_2068/data/1680805782_1681997642/W1681909433_1_CALIB.IMG', # MT3
                       
    # 26
    'COISS_2035/data/1567263044_1567440323/W1567266279_1_CALIB.IMG', # VIO
    'COISS_2016/data/1508882988_1509138645/W1509136315_2_CALIB.IMG', # BL1
    'COISS_2016/data/1508882988_1509138645/W1509136282_2_CALIB.IMG', # GRN
    'COISS_2016/data/1508882988_1509138645/W1509136249_2_CALIB.IMG', # RED
    'COISS_2016/data/1508882988_1509138645/W1509138181_2_CALIB.IMG', # IR1
    'COISS_2016/data/1508882988_1509138645/W1509136446_2_CALIB.IMG', # CB2
    'COISS_2016/data/1508882988_1509138645/W1509137510_2_CALIB.IMG', # CB3
    'COISS_2016/data/1508882988_1509138645/W1509136486_2_CALIB.IMG', # MT2
    'COISS_2016/data/1508882988_1509138645/W1509136099_2_CALIB.IMG', # MT3 
                       
    # 28
    'COISS_2017/data/1514219847_1514299401/W1514281726_1_CALIB.IMG', # VIO
    'COISS_2017/data/1514219847_1514299401/W1514281743_1_CALIB.IMG', # BL1
    'COISS_2017/data/1514219847_1514299401/W1514281760_1_CALIB.IMG', # GRN
    'COISS_2017/data/1514219847_1514299401/W1514281777_1_CALIB.IMG', # RED
    'COISS_2017/data/1514219847_1514299401/W1514284221_1_CALIB.IMG', # IR1
    'COISS_2017/data/1514219847_1514299401/W1514281650_1_CALIB.IMG', # CB3
    'COISS_2017/data/1514219847_1514299401/W1514284552_1_CALIB.IMG', # MT3
    
    # 32
    'COISS_2063/data/1655742537_1655905033/W1655808265_1_CALIB.IMG', # VIO
    'COISS_2063/data/1655742537_1655905033/W1655808298_1_CALIB.IMG', # BL1
    'COISS_2063/data/1655742537_1655905033/W1655808331_1_CALIB.IMG', # GRN
    'COISS_2063/data/1655742537_1655905033/W1655808364_1_CALIB.IMG', # RED
    'COISS_2063/data/1655742537_1655905033/W1655809059_1_CALIB.IMG', # CB2
    'COISS_2063/data/1655742537_1655905033/W1655808656_1_CALIB.IMG', # CB3
    'COISS_2063/data/1655742537_1655905033/W1655809004_1_CALIB.IMG', # MT2
    'COISS_2063/data/1655742537_1655905033/W1655808750_1_CALIB.IMG', # MT3
                           
    # 36
    'COISS_2055/data/1624240547_1624420949/W1624420234_1_CALIB.IMG', # VIO
    'COISS_2055/data/1624240547_1624420949/W1624420267_1_CALIB.IMG', # BL1
    'COISS_2055/data/1624240547_1624420949/W1624420300_1_CALIB.IMG', # GRN
    'COISS_2055/data/1624240547_1624420949/W1624420333_1_CALIB.IMG', # RED
    'COISS_2055/data/1624240547_1624420949/W1624419266_1_CALIB.IMG', # CB2
    'COISS_2055/data/1624240547_1624420949/W1624420657_1_CALIB.IMG', # CB3
    'COISS_2055/data/1624240547_1624420949/W1624419283_1_CALIB.IMG', # MT2
    'COISS_2055/data/1624240547_1624420949/W1624420751_1_CALIB.IMG', # MT3
        
    # 46
    'COISS_2068/data/1683372651_1683615321/W1683615178_1_CALIB.IMG', # VIO
    'COISS_2068/data/1683372651_1683615321/W1683615211_1_CALIB.IMG', # BL1
    'COISS_2068/data/1683372651_1683615321/W1683615266_1_CALIB.IMG', # GRN
    'COISS_2068/data/1683372651_1683615321/W1683615321_1_CALIB.IMG', # RED
    'COISS_2068/data/1683616251_1683727987/W1683616251_1_CALIB.IMG', # CB2
    'COISS_2068/data/1683616251_1683727987/W1683616447_1_CALIB.IMG', # CB3
    'COISS_2068/data/1683616251_1683727987/W1683616313_1_CALIB.IMG', # MT2
    'COISS_2068/data/1683616251_1683727987/W1683616391_1_CALIB.IMG', # MT3
                           
    # 55
    'COISS_2011/data/1492217706_1492344437/W1492341257_2_CALIB.IMG', # VIO
    'COISS_2011/data/1492217706_1492344437/W1492341209_2_CALIB.IMG', # BL1
    'COISS_2011/data/1492217706_1492344437/W1492341176_2_CALIB.IMG', # GRN
    'COISS_2011/data/1492217706_1492344437/W1492341143_2_CALIB.IMG', # RED
    'COISS_2011/data/1492217706_1492344437/W1492341292_2_CALIB.IMG', # CB2
    'COISS_2011/data/1492217706_1492344437/W1492342134_2_CALIB.IMG', # CB3
    'COISS_2011/data/1492217706_1492344437/W1492341332_2_CALIB.IMG', # MT2
    'COISS_2011/data/1492217706_1492344437/W1492341666_2_CALIB.IMG', # MT3
    
    # 60
    'COISS_2030/data/1552197101_1552225837/W1552216646_1_CALIB.IMG', # VIO
    'COISS_2030/data/1552197101_1552225837/W1552216606_1_CALIB.IMG', # BL1
    'COISS_2030/data/1552197101_1552225837/W1552216573_1_CALIB.IMG', # GRN
    'COISS_2030/data/1552197101_1552225837/W1552216540_1_CALIB.IMG', # RED
    'COISS_2030/data/1552197101_1552225837/W1552216381_1_CALIB.IMG', # CB2
    'COISS_2030/data/1552197101_1552225837/W1552216035_1_CALIB.IMG', # CB3
    'COISS_2030/data/1552197101_1552225837/W1552216451_1_CALIB.IMG', # MT2
    'COISS_2030/data/1552197101_1552225837/W1552216219_1_CALIB.IMG', # MT3
                           
    # 68
    'COISS_2030/data/1550711108_1550921975/W1550838842_1_CALIB.IMG', # VIO
    'COISS_2030/data/1550711108_1550921975/W1550838802_1_CALIB.IMG', # BL1
    'COISS_2030/data/1550711108_1550921975/W1550838769_1_CALIB.IMG', # GRN
    'COISS_2030/data/1550711108_1550921975/W1550838736_1_CALIB.IMG', # RED
    'COISS_2030/data/1550711108_1550921975/W1550838572_1_CALIB.IMG', # CB2
    'COISS_2030/data/1550711108_1550921975/W1550838210_1_CALIB.IMG', # CB3
    'COISS_2030/data/1550711108_1550921975/W1550838647_1_CALIB.IMG', # MT2
    'COISS_2030/data/1550711108_1550921975/W1550838394_1_CALIB.IMG', # MT3
                           
    # 73
    'COISS_2082/data/1743902905_1744323160/W1743914297_1_CALIB.IMG', # VIO
    'COISS_2082/data/1743902905_1744323160/W1743914227_1_CALIB.IMG', # BL1
    'COISS_2082/data/1743902905_1744323160/W1743914172_1_CALIB.IMG', # GRN
    'COISS_2082/data/1743902905_1744323160/W1743914117_1_CALIB.IMG', # RED
                           
    # 80
    'COISS_2082/data/1748047605_1748159948/W1748064618_1_CALIB.IMG', # CLEAR
    'COISS_2082/data/1748047605_1748159948/W1748065141_1_CALIB.IMG', # VIO
    'COISS_2082/data/1748047605_1748159948/W1748065093_3_CALIB.IMG', # BL1
    'COISS_2082/data/1748047605_1748159948/W1748065060_1_CALIB.IMG', # GRN
    'COISS_2082/data/1748047605_1748159948/W1748065027_1_CALIB.IMG', # RED
    'COISS_2082/data/1748047605_1748159948/W1748064784_1_CALIB.IMG', # CB3
    'COISS_2082/data/1748047605_1748159948/W1748064690_3_CALIB.IMG', # MT3
                           
    # 85
    'COISS_2085/data/1757258657_1757635754/W1757630543_1_CALIB.IMG', # BL1
    'COISS_2085/data/1757258657_1757635754/W1757630576_1_CALIB.IMG', # GRN
    'COISS_2085/data/1757258657_1757635754/W1757630609_1_CALIB.IMG', # RED
    'COISS_2085/data/1757258657_1757635754/W1757630775_1_CALIB.IMG', # CB3
                           
    # 90
    'COISS_2084/data/1753489519_1753577899/W1753508727_1_CALIB.IMG', # VIO
    'COISS_2084/data/1753489519_1753577899/W1753508760_1_CALIB.IMG', # BL1
    'COISS_2084/data/1753489519_1753577899/W1753508793_1_CALIB.IMG', # GRN
    'COISS_2084/data/1753489519_1753577899/W1753508826_1_CALIB.IMG', # RED
    'COISS_2084/data/1753489519_1753577899/W1753507284_1_CALIB.IMG', # CB2
    'COISS_2084/data/1753489519_1753577899/W1753508679_1_CALIB.IMG', # CB3
    'COISS_2084/data/1753489519_1753577899/W1753507324_1_CALIB.IMG', # MT2
    'COISS_2084/data/1753489519_1753577899/W1753508607_1_CALIB.IMG', # MT3
        
    # 95
    'COISS_2049/data/1604402501_1604469049/W1604460899_1_CALIB.IMG', # VIO
    'COISS_2049/data/1604402501_1604469049/W1604460932_1_CALIB.IMG', # BL1
    'COISS_2049/data/1604402501_1604469049/W1604460965_1_CALIB.IMG', # GRN
    'COISS_2049/data/1604402501_1604469049/W1604460998_1_CALIB.IMG', # RED
    'COISS_2049/data/1604402501_1604469049/W1604461827_1_CALIB.IMG', # CB2
    'COISS_2049/data/1604402501_1604469049/W1604461370_1_CALIB.IMG', # CB3
    'COISS_2049/data/1604402501_1604469049/W1604461794_1_CALIB.IMG', # MT2
    'COISS_2049/data/1604402501_1604469049/W1604461464_1_CALIB.IMG', # MT3
                           
    # 104
    'COISS_2028/data/1548522106_1548756214/W1548712789_1_CALIB.IMG', # VIO
    'COISS_2028/data/1548522106_1548756214/W1548712741_1_CALIB.IMG', # BL1
    'COISS_2028/data/1548522106_1548756214/W1548712708_1_CALIB.IMG', # GRN
    'COISS_2028/data/1548522106_1548756214/W1548712675_1_CALIB.IMG', # RED
    'COISS_2028/data/1548522106_1548756214/W1548716874_1_CALIB.IMG', # IR1
    'COISS_2028/data/1548522106_1548756214/W1548712208_1_CALIB.IMG', # CB3
        
    # 110
    'COISS_2086/data/1760272897_1760532147/W1760465018_1_CALIB.IMG', # VIO
    'COISS_2086/data/1760272897_1760532147/W1760465051_1_CALIB.IMG', # BL1
    'COISS_2086/data/1760272897_1760532147/W1760464970_1_CALIB.IMG', # RED
    'COISS_2086/data/1760272897_1760532147/W1760464915_1_CALIB.IMG', # CB3
                          
    # 114
    'COISS_2030/data/1552128674_1552196996/W1552152072_1_CALIB.IMG', # VIO
    'COISS_2030/data/1552128674_1552196996/W1552152032_1_CALIB.IMG', # BL1
    'COISS_2030/data/1552128674_1552196996/W1552151999_1_CALIB.IMG', # GRN
    'COISS_2030/data/1552128674_1552196996/W1552151966_1_CALIB.IMG', # RED
    'COISS_2030/data/1552128674_1552196996/W1552151900_1_CALIB.IMG', # CB2
    'COISS_2030/data/1552128674_1552196996/W1552151819_1_CALIB.IMG', # CB3
    'COISS_2030/data/1552128674_1552196996/W1552151933_1_CALIB.IMG', # MT2
    'COISS_2030/data/1552128674_1552196996/W1552151867_1_CALIB.IMG', # MT3
        
    # 119
    'COISS_2024/data/1532184650_1532257621/W1532185013_1_CALIB.IMG', # VIO
    'COISS_2024/data/1532184650_1532257621/W1532185046_1_CALIB.IMG', # BL1
    'COISS_2024/data/1532184650_1532257621/W1532184980_1_CALIB.IMG', # GRN
    'COISS_2024/data/1532184650_1532257621/W1532184947_1_CALIB.IMG', # RED
    'COISS_2024/data/1532136934_1532184428/W1532183802_1_CALIB.IMG', # IR1
    'COISS_2024/data/1532184650_1532257621/W1532185079_1_CALIB.IMG', # IR3
    'COISS_2024/data/1532184650_1532257621/W1532184650_1_CALIB.IMG', # IR5
    'COISS_2024/data/1532184650_1532257621/W1532184723_1_CALIB.IMG', # CB2
    'COISS_2024/data/1532136934_1532184428/W1532184214_1_CALIB.IMG', # MT2
    'COISS_2024/data/1532136934_1532184428/W1532183891_1_CALIB.IMG', # MT3
                           
#    # 125
#    'COISS_2015/data/1503416732_1503874914/W1503421986_2_CALIB.IMG', # VIO
#    'COISS_2015/data/1503416732_1503874914/W1503422118_2_CALIB.IMG', # BL1
#    'COISS_2015/data/1503416732_1503874914/W1503422052_2_CALIB.IMG', # GRN
#    'COISS_2015/data/1503416732_1503874914/W1503422085_2_CALIB.IMG', # RED
#                           
#    # 130
#    'COISS_2055/data/1622711732_1623166344/W1622977122_1_CALIB.IMG', # VIO
#    'COISS_2055/data/1622711732_1623166344/W1622977155_1_CALIB.IMG', # BL1
#    'COISS_2055/data/1622711732_1623166344/W1622977188_1_CALIB.IMG', # GRN
#    'COISS_2055/data/1622711732_1623166344/W1622977221_1_CALIB.IMG', # RED
#    'COISS_2055/data/1622711732_1623166344/W1622977738_1_CALIB.IMG', # CB2
#    'COISS_2055/data/1622711732_1623166344/W1622977497_1_CALIB.IMG', # CB3
#    'COISS_2055/data/1622711732_1623166344/W1622977705_1_CALIB.IMG', # MT2
#    'COISS_2055/data/1622711732_1623166344/W1622977567_1_CALIB.IMG', # MT3
#                           
#    # 135
#    'COISS_2060/data/1643317802_1643406946/W1643375992_1_CALIB.IMG', # VIO
#    'COISS_2060/data/1643317802_1643406946/W1643376025_1_CALIB.IMG', # BL1
#    'COISS_2060/data/1643317802_1643406946/W1643376058_1_CALIB.IMG', # GRN
#    'COISS_2060/data/1643317802_1643406946/W1643376091_1_CALIB.IMG', # RED
#    'COISS_2060/data/1643317802_1643406946/W1643376539_1_CALIB.IMG', # CB2
#    'COISS_2060/data/1643317802_1643406946/W1643376327_1_CALIB.IMG', # CB3
#    'COISS_2060/data/1643317802_1643406946/W1643376506_1_CALIB.IMG', # MT2
#    'COISS_2060/data/1643317802_1643406946/W1643376375_1_CALIB.IMG', # MT3
#                           
#    # 140
#    'COISS_2076/data/1727255245_1727449737/W1727330741_1_CALIB.IMG', # VIO
#    'COISS_2076/data/1727255245_1727449737/W1727330774_1_CALIB.IMG', # BL1
#    'COISS_2076/data/1727255245_1727449737/W1727330807_1_CALIB.IMG', # GRN
#    'COISS_2076/data/1727255245_1727449737/W1727330840_1_CALIB.IMG', # RED
#    'COISS_2076/data/1727255245_1727449737/W1727332645_1_CALIB.IMG', # CB2
#    'COISS_2076/data/1727255245_1727449737/W1727332801_1_CALIB.IMG', # CB3
#    'COISS_2076/data/1727255245_1727449737/W1727332572_1_CALIB.IMG', # MT2
#    'COISS_2076/data/1727255245_1727449737/W1727332734_1_CALIB.IMG', # MT3
#                           
#    # 145
#    'COISS_2062/data/1652952601_1653081456/W1652985001_1_CALIB.IMG', # VIO
#    'COISS_2062/data/1652952601_1653081456/W1652985034_1_CALIB.IMG', # BL1
#    'COISS_2062/data/1652952601_1653081456/W1652985067_1_CALIB.IMG', # GRN
#    'COISS_2062/data/1652952601_1653081456/W1652985100_1_CALIB.IMG', # RED
#    'COISS_2062/data/1652952601_1653081456/W1652985344_1_CALIB.IMG', # CB3
#    'COISS_2062/data/1652952601_1653081456/W1652985414_1_CALIB.IMG', # MT3
#                           
#    # 149
#    'COISS_2076/data/1721802517_1721894741/W1721822901_1_CALIB.IMG', # CLEAR
#    'COISS_2076/data/1721802517_1721894741/W1721822934_1_CALIB.IMG', # VIO
#    'COISS_2076/data/1721802517_1721894741/W1721822967_1_CALIB.IMG', # BL1
#    'COISS_2076/data/1721802517_1721894741/W1721823000_1_CALIB.IMG', # GRN
#    'COISS_2076/data/1721802517_1721894741/W1721823033_1_CALIB.IMG', # RED
#    'COISS_2076/data/1721802517_1721894741/W1721823066_1_CALIB.IMG', # IR1
#    'COISS_2076/data/1721802517_1721894741/W1721823382_1_CALIB.IMG', # IR2
#    'COISS_2076/data/1721802517_1721894741/W1721823099_1_CALIB.IMG', # IR3
#    'COISS_2076/data/1721802517_1721894741/W1721823316_1_CALIB.IMG', # CB2
#    'COISS_2076/data/1721802517_1721894741/W1721823147_1_CALIB.IMG', # CB3
#    'COISS_2076/data/1721802517_1721894741/W1721823349_1_CALIB.IMG', # MT2
#    'COISS_2076/data/1721802517_1721894741/W1721823283_1_CALIB.IMG', # MT3
#                           
#    # 155
#    'COISS_2074/data/1717673485_1717756625/W1717682790_1_CALIB.IMG', # VIO
#    'COISS_2074/data/1717673485_1717756625/W1717688223_1_CALIB.IMG', # BL1
#    'COISS_2074/data/1717673485_1717756625/W1717682856_1_CALIB.IMG', # GRN
#    'COISS_2074/data/1717673485_1717756625/W1717688289_1_CALIB.IMG', # RED
#    'COISS_2074/data/1717673485_1717756625/W1717688370_1_CALIB.IMG', # MT3
#                           
#    # 158
#    'COISS_2009/data/1487140530_1487182149/W1487166992_2_CALIB.IMG', # CLEAR
#    'COISS_2009/data/1487140530_1487182149/W1487167364_2_CALIB.IMG', # VIO
#    'COISS_2009/data/1487140530_1487182149/W1487167243_2_CALIB.IMG', # BL1
#    'COISS_2009/data/1487140530_1487182149/W1487167154_2_CALIB.IMG', # GRN
#    'COISS_2009/data/1487140530_1487182149/W1487167033_2_CALIB.IMG', # RED
#        
#    # 164
#    'COISS_2074/data/1716194058_1716328931/W1716307950_1_CALIB.IMG', # VIO
#    'COISS_2074/data/1716194058_1716328931/W1716307983_1_CALIB.IMG', # BL1
#    'COISS_2074/data/1716194058_1716328931/W1716308016_1_CALIB.IMG', # GRN
#    'COISS_2074/data/1716194058_1716328931/W1716308049_1_CALIB.IMG', # RED
#                           
#    # 166
#    'COISS_2033/data/1561668355_1561837358/W1561790952_1_CALIB.IMG', # CLEAR
#    'COISS_2033/data/1561668355_1561837358/W1561794086_1_CALIB.IMG', # VIO
#    'COISS_2033/data/1561668355_1561837358/W1561794119_1_CALIB.IMG', # BL1
#    'COISS_2033/data/1561668355_1561837358/W1561794152_1_CALIB.IMG', # GRN
#    'COISS_2033/data/1561668355_1561837358/W1561794185_1_CALIB.IMG', # RED
#    'COISS_2033/data/1561668355_1561837358/W1561794349_1_CALIB.IMG', # MT2
]


FILTER_COLOR = {
    'CLEAR': (.7,.7,.7),
    
    'RED': (1,0,0),
    'GRN': (0,1,0),
    'BL1': (0,0,1),
    'VIO': (159/256.,0,1),
    
    'MT2': (1,204/256.,153/256.),
    'CB2': (1,.5,0),

    'MT3': (1,153/256.,204/256.),
    'CB3': (1,.3,1),
}    

    
#===============================================================================
# 
# CREATE METADATA
#
#===============================================================================

def shift_image(image, offset_u, offset_v):
    """Shift an image by an offset.
    
    Pad the new area with zero and throw away the data moved off the edge."""
    if offset_u == 0 and offset_v == 0:
        return image
    
    image = np.roll(image, -offset_u, 1)
    image = np.roll(image, -offset_v, 0)

    if offset_u != 0:    
        if offset_u < 0:
            image[:,:-offset_u] = 0
        else:
            image[:,-offset_u:] = 0
    if offset_v != 0:
        if offset_v < 0:
            image[:-offset_v,:] = 0
        else:
            image[-offset_v:,:] = 0
    
    return image

def make_bins(full_incidence, incidence_res, full_emission, emission_res):
    i_bins = (full_incidence / incidence_res).astype('int')
    e_bins = (full_emission / emission_res).astype('int')

    num_incidence_bins = int(np.ceil(360. / incidence_res))
    num_emission_bins = int(np.ceil(1. / emission_res))
    
    bins = []
    num_bins = 0
    for i in xrange(num_incidence_bins):
        bins.append([])
        for j in xrange(num_emission_bins):
            bins[-1].append([])
            num_bins += 1
        
    return i_bins, e_bins, bins
    
def bin_one_image(i_bins, e_bins, data, full_mask, bins, u_offset, v_offset, verbose=False):
    bins = copy.deepcopy(bins)
    num_incidence_bins = len(bins)
    num_emission_bins = len(bins[0])

    full_mask = shift_image(full_mask, u_offset, v_offset)
    full_data = data[full_mask].flatten()

    if full_data.shape != i_bins.shape:
        if verbose:
            print 'Offset shifts off edge'
            return 1e38, 0
        
    if verbose:
        print 'Populating bins'
            
    for i_bin, e_bin, datum in zip(i_bins, e_bins, full_data):
        bins[i_bin][e_bin].append(datum)
    
    if verbose:
        print 'Computing standard deviations'
    
    std_list = []
    for i in xrange(num_incidence_bins):
        for j in xrange(num_emission_bins):
            if len(bins[i][j]) > 1:
                std_list.append(np.std(bins[i][j]))
    
    rms = np.sqrt(np.sum(np.array(std_list)**2))
    
    return rms, len(std_list)
    
def find_offset_one_image(filename, save=True, display=True, verbose=True):
    print 'Finding offset for', filename

    full_filename = os.path.join(COISS_2XXX_DERIVED_ROOT, filename)
    obs = read_iss_file(full_filename)
    orig_fov = obs.fov

    best_offset = None
    best_rms = 1e38
    best_overlay = None
    orig_overlay = None

    bp = oops.Backplane(obs)
    
    if verbose:
        print 'Computing backplanes'
    
    bp_incidence = bp.incidence_angle('TITAN+ATMOSPHERE') * oops.DPR
    bp_emission = bp.emission_angle('TITAN+ATMOSPHERE').sin()# * oops.DPR
    bp_incidence.vals[bp_incidence.mask] = 0.
    bp_emission.vals[bp_incidence.mask] = 0.
    
#    plt.imshow(bp_incidence.vals)
#    plt.figure()
#    plt.imshow(bp_emission.vals)
#    plt.show()
    
    full_mask = np.logical_not(bp_incidence.mask)
    full_mask[bp_emission.vals < 0.85] = 0
    full_mask[:5,:] = 0
    full_mask[-5:,:] = 0
    full_mask[:,:5] = 0
    full_mask[:,-5:] = 0
            
    full_incidence = bp_incidence.vals[full_mask].flatten()
    full_emission  = bp_emission.vals[full_mask].flatten()

    area = np.sum(full_mask)
    diameter = np.sqrt(area / oops.PI) * 2
    
    i_res = 360. / diameter
    e_res = 1. / diameter
    
    print 'Choosing i_res', i_res, 'e_res', e_res

    i_bins, e_bins, bins = make_bins(full_incidence, i_res,
                                     full_emission, e_res)
    
    if False:
        # Find the optimal resolutions
    
        if verbose:
            print 'Finding optimal resolutions'
    
        best_i_res = None
        best_e_res = None
        best_num_bins = 0
        
        for i_res in np.arange(0.3, 5., 0.3):
            for e_res in np.arange(0.002, 0.05, 0.002):
                rms, num_bins = bin_one_image(full_incidence, i_res, 
                                              full_emission, e_res,
                                              obs.data, full_mask, 0, 0,
                                              verbose=True)
                print i_res, e_res, num_bins
                if num_bins > best_num_bins:
                    best_num_bins = num_bins
                    best_i_res = i_res
                    best_e_res = e_res
        
        print 'BEST', best_i_res, best_e_res
        assert False

    
    limit = 20
    
    for u_offset in xrange(-limit,limit+1):
        for v_offset in xrange(-limit,limit+1):
            if verbose:
                print 'Trying offset', u_offset, v_offset
    
            rms, num_bins = bin_one_image(i_bins, e_bins, obs.data, full_mask,
                                          bins, u_offset, v_offset, 
                                          verbose=True)
            
            if verbose:
                print 'GOOD BINS', num_bins, 'RMS STD', rms
            
            if rms < best_rms:
                best_rms = rms
                best_offset = (u_offset, v_offset)

    print 'FINAL RESULT', best_offset, best_rms

    if display:
        orig_overlay = np.zeros(obs.data.shape+(3,))
        orig_overlay[:,:,0] = full_mask
        best_overlay = np.zeros(obs.data.shape+(3,))
        best_overlay[:,:,0] = shift_image(full_mask, best_offset[0], best_offset[1])

        im = imgdisp.ImageDisp([obs.data, obs.data], [orig_overlay, best_overlay],
                               canvas_size=None,
                               title=filename, allow_enlarge=True,
                               one_zoom=True, auto_update=True)
        tk.mainloop()

            
     
def backplane_one_image(filename_list, save=True, display=False, offset=None):
    print 'Analyzing', filename_list
    if type(filename_list) != type((0,)):
        filename_list = [filename_list]

    full_phase = np.array([])
    full_incidence = np.array([])
    full_emission = np.array([])
    full_data = np.array([])
    
    for filename in filename_list:
        full_filename = os.path.join(COISS_2XXX_DERIVED_ROOT, filename)
        obs = read_iss_file(full_filename)
#        offset_metadata = file_read_offset_metadata(full_filename)
#    
#        if offset_metadata is None:
#            print 'Warning: No offset data for', filename
#        else:
#            offset = offset_metadata['offset']
        if offset:
            obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
            print 'Using offset', offset
    
        bp = oops.Backplane(obs)
        
        phase = bp.phase_angle('TITAN+ATMOSPHERE')
        incidence = bp.incidence_angle('TITAN+ATMOSPHERE')
        emission = bp.emission_angle('TITAN+ATMOSPHERE')

        full_mask = np.logical_not(phase.mask)
        full_mask[:5,:] = 0
        full_mask[-5:,:] = 0
        full_mask[:,:5] = 0
        full_mask[:,-5:] = 0
        
        full_phase     = np.append(full_phase,
                                   phase.vals[full_mask].flatten())
        full_incidence = np.append(full_incidence,
                                   incidence.vals[full_mask].flatten())
        full_emission  = np.append(full_emission,
                                   emission.vals[full_mask].flatten())
        full_data      = np.append(full_data,
                                   obs.data[full_mask].flatten())

        if display:
            overlay = np.zeros(obs.data.shape+(3,))
            overlay[:,:,0] = full_mask
            
            im = imgdisp.ImageDisp([obs.data], [overlay],
                                   canvas_size=None,
                                   title=filename, allow_enlarge=True,
                                   one_zoom=True, auto_update=True)
            tk.mainloop()
            
    mean_phase = np.mean(full_phase) * oops.DPR
    print obs.data.shape, obs.filter1, obs.filter2,
        
    clean_name = file_clean_name(filename_list[0])
    ph_name = ('%6.2f' % mean_phase) + '_' + clean_name

    print 'Mean phase %6.2f' % mean_phase,
    print ph_name
    
    if save:
        np.savez(os.path.join(RESULTS_ROOT, 'titan', clean_name),
                 full_phase=full_phase, full_incidence=full_incidence, 
                 full_emission=full_emission, full_data=full_data)
     
def backplane_all_images(force=False):
    for filename in IMAGE_LIST_BY_PHASE:
        if force:
            backplane_one_image(filename)
        else:
            try:
                if type(filename) == type((0,)):
                    clean_name = file_clean_name(filename[0])
                else:
                    clean_name = file_clean_name(filename)
                np_arrays = np.load(os.path.join(RESULTS_ROOT, 'titan', 
                                                 clean_name+'.npz'))
            except IOError:
                backplane_one_image(filename)


#===============================================================================
#
# READ METADATA
# 
#===============================================================================

DATA_BY_FILTER = {}
DATA_BY_PHASE = {}

def read_backplanes():
    for filename in IMAGE_LIST_BY_PHASE:
        if type(filename) == type((0,)):
            filename = filename[0]
        print 'Reading', filename

        clean_name = file_clean_name(filename)
        
        try:
            np_arrays = np.load(os.path.join(RESULTS_ROOT, 'titan', clean_name+'.npz'))
        except IOError:
            print 'NO DATA FILE', clean_name
            continue

        full_filename = os.path.join(COISS_2XXX_DERIVED_ROOT, filename)
        obs = read_iss_file(full_filename)
        
        full_data = np_arrays['full_data']
        full_phase = np_arrays['full_phase']
        full_incidence = np_arrays['full_incidence']
        full_emission = np_arrays['full_emission']
        mean_phase = np.mean(full_phase) * oops.DPR

        if obs.filter1 == 'CL1' and obs.filter2 == 'CL2':
            filter = 'CLEAR'
        else:
            filter = obs.filter1
            if filter == 'CL1':
                filter = obs.filter2
            elif obs.filter2 != 'CL2':
                filter += '+' + obs.filter2
                
        if filter not in DATA_BY_FILTER:
            DATA_BY_FILTER[filter] = {}

        data = (clean_name, full_data, full_phase, full_incidence, 
                full_emission)
        
        DATA_BY_FILTER[filter][mean_phase] = data
        DATA_BY_PHASE[(mean_phase, filter)] = data 


#===============================================================================
#
# PLOTTING HELPERS
# 
#===============================================================================
    
#def show_plots(name=None, use_minnaert=True):
#    for filename in sorted(DISC_DATA):
#        print filename, name
#        if name is not None and filename != name:
#            continue
#        full_data = FULL_DATA[filename]
#        full_phase = FULL_PHASE[filename]
#        full_incidence = FULL_INCIDENCE[filename]
#        full_emission = FULL_EMISSION[filename]
#
#        if use_minnaert:
#            cutoff = 80 * oops.RPD
#            
#            odata = full_data[np.where(full_incidence < cutoff)]
#            oemission = full_emission[np.where(full_incidence < cutoff)]
#            ophase = full_phase[np.where(full_incidence < cutoff)]
#            oincidence = full_incidence[np.where(full_incidence < cutoff)]
#            mean_phase = np.mean(full_phase) * oops.DPR
#            print 'Mean phase', mean_phase
#            if mean_phase > 108:
#                continue
#            
#            mi, me = sciopt.fmin(minnaert_opt_func, (0.5,0.5),
#                                 args=(oincidence, oemission, odata),
#                                 disp=0)
#
#        plot_one(full_data, full_phase, full_incidence, full_emission)
#        plt.title(filename)
#        plt.show()

def show_plots_multiple_filters(filters=None, phases=None):
    phase_tolerance = 2.
    if phases is None:
        phases = np.arange(phase_tolerance, 360, phase_tolerance*2)
        
    for goal_phase in phases:
        first_one = True
        used_filters = []
        # For each filter, find the phase entry closest to the target phase
        for filter in sorted(DATA_BY_FILTER):
            if filters and filter not in filters:
                continue
        
            best_phase = None
            best_phase_dist = 1e38
            for phase in DATA_BY_FILTER[filter]:
                delta = abs(goal_phase-phase)
                if delta < phase_tolerance:
                    if delta < best_phase_dist:
                        best_phase_dost = delta
                        best_phase = phase 

            if best_phase is None:
                continue
        
            print 'FOUND', filter, 'GOAL', goal_phase, 'ACTUAL', best_phase
            
            if first_one:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                first_one = False
        
            (clean_name, full_data, full_phase, full_incidence, 
             full_emission) = DATA_BY_PHASE[(best_phase,filter)]
                
            used_filters.append(filter)

            full_decimate = 1#max(1, len(full_data) // 10000)
            full_decimate = 100
                    
            full_data = full_data[::full_decimate]
            full_phase = full_phase[::full_decimate]
            full_incidence = full_incidence[::full_decimate]
            full_emission = full_emission[::full_decimate]

            if filter not in FILTER_COLOR:
                print 'Unknown filter', filter, 'file', clean_name
                continue
            
            base_color = FILTER_COLOR[filter]

            colors = []
            for data_idx in xrange(len(full_data)):
                shade = np.cos(full_emission[data_idx])
                color = (base_color[0] * shade,
                         base_color[1] * shade,
                         base_color[2] * shade)
                colors.append(color)
            plt.scatter(full_incidence*oops.DPR,
                        full_data, s=20, c=colors, alpha=0.05,
                        edgecolors='none')
        
        if first_one:
            continue
        
        print '-------------'
        
        xlimits = plt.xlim()
        ylimits = plt.ylim()
        
        for filter in sorted(used_filters):
            if filter not in FILTER_COLOR:
                continue
            plt.plot(-100,-100, 'o', mec=FILTER_COLOR[filter], mfc=FILTER_COLOR[filter], ms=10,
                     label=filter)

        plt.xlim(0, xlimits[1])
        plt.ylim(0, ylimits[1])
        
        plt.legend(numpoints=1)
        plt.xlabel('Incidence angle')
        plt.ylabel('I/F')
        plt.title('PHASE %7.2f' % goal_phase)
        plt.show()
        

def show_plots_multiple_phases(filters=None, phases=None):
    phase_tolerance = 2.
    if phases is None:
        phases = np.arange(phase_tolerance, 360, phase_tolerance*2)
        
    for filter in sorted(DATA_BY_FILTER):
        if filters and filter not in filters:
            continue

        phase_list = []
        
        for goal_phase in phases:
            first_one = True
            used_images = []

            for phase in sorted(DATA_BY_FILTER[filter]):
                delta = abs(goal_phase-phase)
                if delta < phase_tolerance:
                    phase_list.append(phase)               
                    print 'FOUND', filter, 'GOAL', goal_phase, 'ACTUAL', phase

            if len(phase_list) == 0:
                continue
                    
            fig = plt.figure()
            ax = fig.add_subplot(111)
            first_one = False
            
            for phase_num, phase in enumerate(phase_list):
                (clean_name, full_data, full_phase, full_incidence, 
                 full_emission) = DATA_BY_PHASE[(phase,filter)]
                    
                used_images.append(clean_name)
    
                full_decimate = 1#max(1, len(full_data) // 10000)
#                full_decimate = 100
                        
                full_data = full_data[::full_decimate]
                full_phase = full_phase[::full_decimate]
                full_incidence = full_incidence[::full_decimate]
                full_emission = full_emission[::full_decimate]

                base_color = colorsys.hsv_to_rgb(
                         float(phase_num)/len(phase_list), 1, 1)
    
                colors = []
                for data_idx in xrange(len(full_data)):
                    shade = np.cos(full_emission[data_idx])
                    shade = 1
                    color = (base_color[0] * shade,
                             base_color[1] * shade,
                             base_color[2] * shade)
                    colors.append(color)
                plt.scatter(full_incidence*oops.DPR,
                            full_data, s=20, c=colors, alpha=0.05,
                            edgecolors='none')
        
        if first_one:
            continue
        
        print '-------------'
        
        xlimits = plt.xlim()
        ylimits = plt.ylim()
        
        for image_num, image in enumerate(used_images):
            color = colorsys.hsv_to_rgb(
                            float(image_num)/len(phase_list), 1, 1)
            plt.plot(-100,-100, 'o', ms=10,
                     mec=color, mfc=color, 
                     label=('%s %7.3f' % (image, phase_list[image_num])))

        plt.xlim(0, xlimits[1])
        plt.ylim(0, ylimits[1])
        
        plt.legend(numpoints=1)
        plt.xlabel('Incidence angle')
        plt.ylabel('I/F')
        plt.title('PHASE %7.2f' % goal_phase)
        plt.show()
        
        
def plot_one(disc_data, disc_phase, disc_incidence, disc_emission, 
             full_data=None, full_phase=None, full_incidence=None, 
             full_emission=None,
             phot_func=None):
    disc_decimate = len(disc_data) // 1000
    full_decimate = 100
            
#    disc_decimate = 10
    
    disc_data = disc_data[::disc_decimate]
    disc_phase = disc_phase[::disc_decimate]
    disc_incidence = disc_incidence[::disc_decimate]
    disc_emission = disc_emission[::disc_decimate]
    
    phase_list = [1]
    if full_data:
        full_data = full_data[::full_decimate]
        full_phase = full_phase[::full_decimate]
        full_incidence = full_incidence[::full_decimate]
        full_emission = full_emission[::full_decimate]
        phase_list = [0,1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for phase in phase_list:
        if phase == 0:
            incidence = disc_incidence
            emission = disc_emission
            data = disc_data
            marker = '.'
        else:
            incidence = full_incidence
            emission = full_emission
            data = full_data
            marker = '+'
        color_by_vals = np.cos(emission)
        min_v = 0.
        max_v = 1.
        for idx in xrange(len(incidence)):
            i = incidence[idx]
            e = emission[idx]
            color_by_val = color_by_vals[idx]

            phot = 1.
            if phot_func:
                phot = phot_func(np.array([i]), np.array([e]))
            iof = data[idx] / phot
            color_by = (color_by_val-min_v)/(max_v-min_v)
            color = colorsys.hsv_to_rgb(color_by/2+.5, 1, 1)
            plt.plot(i*oops.DPR, iof, marker, color=color, alpha=0.3)

        max_i = np.max(incidence)
        c_range = np.arange(min_v, max_v, (max_v-min_v)/100)
        for idx in xrange(len(c_range)):
            color_by_val = c_range[idx]
            color_by = (color_by_val-min_v)/(max_v-min_v)
            color = colorsys.hsv_to_rgb(color_by/2+.5, 1, 1)
            plt.plot(max_i+10, c_range[idx], '.', color=color)

    if phot_func:
        plt.ylim(0,2)
    else:
        plt.ylim(0,1)
    plt.xlabel('Incidence angle')
    plt.ylabel('I/F')


#===============================================================================
#
# OPTIMIZATION FUNCTIONS
# 
#===============================================================================

# Single Minnaert
#    abs(cos(i))**mi * abs(cos(e))**me

def minnaert_angle_conv(params):
    return params

def minnaert_func(params, incidence, emission):
    a, mi, me = params
    phot = a*(np.abs(np.cos(incidence))**mi *
              np.abs(np.cos(emission))**me)

    print 'SM A %9.6f MI %9.6f ME %9.6f' % (a, mi, me)

    zero_incidence = np.arccos(0)
    phot[np.where(incidence >= zero_incidence)] = 0.
    
    return phot

def minnaert_opt_func(params, incidence, emission, data):
    phot = minnaert_func(params, incidence, emission)

    resid = np.sqrt(np.sum((data-phot)**2))
    print 'RESID', resid
    return resid


# Double Minnaert version 1
#    Before break incidence angle:
#      a1*abs(cos(b1*i+c1))**d1 * cos(e)**d3
#    After break incidence angkle:
#      a2*(abs(cos(b2*i+c2))**d2-1) * cos(e)**d3

def double_minnaert1_angle_conv(params, min_incidence):
    brk_angle, a1_angle, b1_angle, c1_angle, d1_angle, b2_angle, c2_angle, d2_angle, d3_angle = params
    a1 = (np.sin(a1_angle)+1)/2 # 0-1
    b1 = np.sin(b1_angle)/2+1 # .5-1.5
    c1 = np.sin(c1_angle)*oops.PI/4 # -PI/4 - PI/4
    d1 = np.sin(d1_angle)+1 # 0-2
    b2 = -(np.sin(b2_angle)/2+1) # -1.5 - -.5
    c2 = (np.sin(c2_angle)+1)*oops.TWOPI # 0 - FOURPI
    d2 = np.sin(d2_angle)+1 # 0-2
    d3 = 2*np.sin(d3_angle) # -2 - 2
    
    zero_incidence1 = (oops.HALFPI-c1)/b1
    
    # Breakpoint has to be between the minimum incidence angle and the 0-zero crossing of first curve 
    brk = (np.sin(brk_angle)+1)/2 * (zero_incidence1-min_incidence) + min_incidence 
    
    return brk, a1, b1, c1, d1, b2, c2, d2, d3
    
def double_minnaert1_func(params, incidence, emission):
    brk, a1, b1, c1, d1, b2, c2, d2, d3 = double_minnaert1_angle_conv(params,
                                                                 np.min(incidence))
    
    brk_val1 = a1*np.abs(np.cos(b1*brk+c1))**d1 * np.cos(emission)**d3
    brk_val2 = (np.abs(np.cos(b2*brk+c2))**d2-1) * np.cos(emission)**d3
    a2 = brk_val1/brk_val2
    
    phot = a1*np.abs(np.cos(b1*incidence+c1))**d1 * np.cos(emission)**d3
    phot2 = a2*(np.abs(np.cos(b2*incidence+c2))**d2-1) * np.cos(emission)**d3
    phot[np.where(incidence >= brk)] = phot2[np.where(incidence >= brk)]
    
    zero_incidence2 = -(c2%oops.TWOPI)/b2
    while zero_incidence2 > brk-oops.PI/b2:
        zero_incidence2 += oops.PI/b2
    
    phot[np.where(incidence >= zero_incidence2)] = 0.
    
    print 'DM1 BRK %6.2f A1 %7.5f B1 %7.5f C1 %9.6f D1 %7.5f B2 %8.5f C2 %9.6f D2 %7.5f D3 %7.5f' % (
                 brk*oops.DPR, a1, b1, c1, d1, b2, c2, d2, d3)
    
    return phot

def double_minnaert1_opt_func(params, incidence, emission, data):
    phot = double_minnaert1_func(params, incidence, emission)
    resid = np.sqrt(np.sum((data-phot)**2))
    print 'RESID', resid
    return resid


# Double Minnaert version 2
#    a*abs(cos(b*i+c)+e)**d * cos(em)**f

def double_minnaert2_angle_conv(params):
    a_angle, b_angle, c_angle, d_angle, e_angle, f_angle = params
    a = (np.sin(a_angle)+1)/2 # 0-1
    b = 2*np.sin(b_angle)+2.1 # .1-4.1
    c = -(np.sin(c_angle)+1)/2*oops.HALFPI # -PI - PI
#    while c > 0: c = c-oops.PI
#    while c < -oops.PI: c = c+oops.PI
    d = 1.5*(np.sin(d_angle)+1) # 0-6
    e = (np.sin(e_angle)+1)/4 # 0 - 0.5
    f = np.sin(f_angle)-1 # -1 - 1
    
    return a, b, c, d, e, f
    
def double_minnaert2_func(params, incidence, emission, cvt_angle=True):
    if cvt_angle:
        a, b, c, d, e, f = double_minnaert2_angle_conv(params)
    else:
        a, b, c, d, e, f = params
    
    phot = a*np.abs(np.cos(b*(incidence+c))+e)**d * np.cos(emission)**f

    zero_incidence = np.arccos(-e)/b-c
    if zero_incidence < 0:
        zero_incidence += oops.PI
    if zero_incidence > oops.PI:
        zero_incidence -= oops.PI
    phot[incidence >= zero_incidence] = 0.
    
#    print 'DM2 A %9.6f B %9.6f C %9.6f D %9.6f E %9.6f F %9.6f' % (a, b, c, d, e, f)
    
    return phot

def double_minnaert2_opt_func(params, incidence, emission, data):
    phot = double_minnaert2_func(params, incidence, emission)
    resid = np.sqrt(np.sum((data-phot)**2))
#    print 'RESID', resid
    return resid
    

# Arctan version 1
#    a*(arctan(b*i+c)+d) * cos(em)**e

def arctan1_angle_conv(params):
    a_angle, b_angle, c_angle, d_angle, e_angle = params
    a = (np.sin(a_angle)+1)/2 # 0-1
    b = (np.sin(b_angle)-1) # -2 to 0
    c = np.sin(c_angle)*oops.PI # -PI - PI
#    d = 3*(np.sin(d_angle)+1) # 0-3
    d = np.sin(d_angle)*oops.HALFPI # -PI/2 - PI/2
    e = 2*np.sin(e_angle) # -2 - 2
    
    return a, b, c, d, e
    
def arctan1_func(params, incidence, emission):
    a, b, c, d, e = arctan1_angle_conv(params)
    
    phot = a*(np.arctan(b*incidence+c)+d) * np.cos(emission)**e

    print 'AT1 A %9.6f B %9.6f C %9.6f D %9.6f E %9.6f' % (a, b, c, d, e)
    
    return phot

def arctan1_opt_func(params, incidence, emission, data):
    phot = arctan1_func(params, incidence, emission)
    resid = np.sqrt(np.sum((data-phot)**2))
    print 'RESID', resid
    return resid

def get_funcs(alg):
    if alg == 'sm':
        phot_func = minnaert_func
        opt_func = minnaert_opt_func
        angel_func = minnaert_angle_conv
        guess = (0.25,0.5,-0.5)
    elif alg == 'dm1':
        phot_func = double_minnaert1_func
        opt_func = double_minnaert1_opt_func
        angle_func = double_minnaert1_angle_conv
        guess = (oops.PI*.75, 
                 np.arcsin(0.25*2-1), 0., 0., 0.,
                 0., np.arcsin(2.7*oops.PI/oops.TWOPI-1), 0.,
                 0.)    
    elif alg == 'dm2':
        phot_func = double_minnaert2_func
        opt_func = double_minnaert2_opt_func
        angle_func = double_minnaert2_angle_conv
        guess = (np.arcsin(0.25*2-1), 0., 0., 0., 0., 0.)
    elif alg == 'arctan1':
        phot_func = arctan1_func
        opt_func = arctan1_opt_func
        angle_func = arctan1_angle_convguess = (np.arcsin(0.25*2-1), 0., 0., 0., 0.)
    else:
        print 'Unknown algorithm', alg
        assert False

    return phot_func, opt_func, angle_func, guess

def optimize_disc(alg, name=None):
    assert False # Convert to full?
    phot_func, opt_func, angle_func, guess = get_funcs(alg)
    
    for filename in sorted(DISC_DATA):
        print filename, name
        if name is not None and filename != name:
            continue
        print 'Optimizing', alg, filename

        disc_data = DISC_DATA[filename]
        disc_phase = DISC_PHASE[filename]
        disc_incidence = DISC_INCIDENCE[filename]
        disc_emission = DISC_EMISSION[filename]
        
        mean_phase = np.mean(full_phase) * oops.DPR
        print 'Mean phase', mean_phase

        opt_data = disc_data
        opt_phase = disc_phase
        opt_incidence = disc_incidence
        opt_emission = disc_emission
        
        if alg == 'sm':
            cutoff = 80 * oops.RPD        
            opt_data = opt_data[np.where(opt_incidence < cutoff)]
            opt_emission = opt_emission[np.where(opt_incidence < cutoff)]
            opt_phase = opt_phase[np.where(opt_incidence < cutoff)]
            opt_incidence = opt_incidence[np.where(opt_incidence < cutoff)]
            
        opt_decimate = len(opt_data) // 1000
        
        dec_data = opt_data[::opt_decimate]
        dec_phase = opt_phase[::opt_decimate]
        dec_incidence = opt_incidence[::opt_decimate]
        dec_emission = opt_emission[::opt_decimate]

        ret = sciopt.fmin_powell(opt_func, guess,
                                 args=(dec_incidence, dec_emission, dec_data),
                                 disp=0)
        
        plot_one(disc_data, disc_phase, disc_incidence, disc_emission)

        di_incidence = np.arange(0, 180., 0.1) * oops.RPD
        di_zeros = np.zeros(di_incidence.shape)

        for e in [0, 45, 60]:        
            phot = phot_func(ret, di_incidence, di_zeros+e*oops.RPD)        
            color = colorsys.hsv_to_rgb(np.cos(e*oops.RPD)/2+.5, 1, 1)
            plt.plot(di_incidence*oops.DPR, phot, '-', color=color, lw=1)

        plt.title(filename)
        plt.show()

        plot_one(disc_data, disc_phase, disc_incidence, disc_emission,
                 phot_func=lambda i,e: phot_func(ret, i, e))
        plt.title(filename)
        plt.show()

def plot_params(alg):
    assert False # Convert to full?
    phase_list = []
    ret_list = []
    
    phot_func, opt_func, angle_func, guess = get_funcs(alg)
    
    for filename in sorted(DISC_DATA):
        print 'Optimizing', alg, filename

        disc_data = DISC_DATA[filename]
        disc_phase = DISC_PHASE[filename]
        disc_incidence = DISC_INCIDENCE[filename]
        disc_emission = DISC_EMISSION[filename]

        mean_phase = np.mean(full_phase) * oops.DPR
        print 'Mean phase', mean_phase
        
        if mean_phase > 100:
            continue
        if mean_phase < 18:
            continue
        
        opt_data = disc_data
        opt_phase = disc_phase
        opt_incidence = disc_incidence
        opt_emission = disc_emission
        
        if alg == 'sm':
            cutoff = 80 * oops.RPD        
            opt_data = opt_data[np.where(opt_incidence < cutoff)]
            opt_emission = opt_emission[np.where(opt_incidence < cutoff)]
            opt_phase = opt_phase[np.where(opt_incidence < cutoff)]
            opt_incidence = opt_incidence[np.where(opt_incidence < cutoff)]

        opt_decimate = max(len(opt_data) // 10000, 1)
                
        dec_data = opt_data[::opt_decimate]
        dec_phase = opt_phase[::opt_decimate]
        dec_incidence = opt_incidence[::opt_decimate]
        dec_emission = opt_emission[::opt_decimate]

        ret = sciopt.fmin_powell(opt_func, guess,
                                 args=(dec_incidence, dec_emission, dec_data),
                                 disp=0)

        real_ret = angle_func(ret)

        for j in xrange(len(real_ret)):
            print '%9.6f' % real_ret[j],
        print

        phase_list.append(mean_phase)
        ret_list.append(real_ret)
    
    print
    data_lists = []
    for i in xrange(len(ret_list[0])):
        data_lists.append([])
        
    for i in xrange(len(phase_list)):
        print 'PH %7.3f' % phase_list[i],
        for j in xrange(len(ret_list[i])):
            data_lists[j].append(ret_list[i][j])
            print '%9.6f' % ret_list[i][j],
        print
    
    for i in xrange(len(data_lists)):
        print np.polyfit(phase_list, data_lists[i], 1)
        print np.min(data_lists[i]), np.max(data_lists[i]), np.mean(data_lists[i]), np.std(data_lists[i])
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in xrange(len(ret_list[0])):
        plt.plot(phase_list, [x[i] for x in ret_list],
                 '-', label=str(i+1))
        
    plt.legend()
        
    plt.show()

def plot_canned_result(name=None):
    assert False # Convert to full?
    orig_phot_func, opt_func, angle_func, guess = get_funcs('dm2')
    
    phot_func = lambda params, incidence, emission: orig_phot_func(params, incidence, emission, cvt_angle=False)
    
    for filename in sorted(DISC_DATA):
        print filename, name
        if name is not None and filename != name:
            continue
        print 'Displaying', filename

        disc_data = DISC_DATA[filename]
        disc_phase = DISC_PHASE[filename]
        disc_incidence = DISC_INCIDENCE[filename]
        disc_emission = DISC_EMISSION[filename]
        
        mean_phase = np.mean(full_phase) * oops.DPR
        print 'Mean phase', mean_phase

#[-0.00091367  0.20465154]
#0.116122236394 0.207111387164 0.153182435836 0.025327772196
#[ 0.00170376  1.4196773 ]
#1.2941870853 1.65744621641 1.51565424633 0.0878671138453
#[-0.01025439  0.19890163]
#-0.834762884324 -0.00708461479878 -0.378752904923 0.235314820405
#[ 0.02121966  0.10946198]
#0.695123783062 2.22827531327 1.30481719104 0.512535770118
#[-0.0032684   0.58933541]
#0.164388199373 0.498322546288 0.405218424073 0.0943748624381
#[-0.00359419 -0.0005182 ]
#-0.505006529992 -0.000251161577628 -0.202987511611 0.12036164079

        ret = [0.153182435836, # A
               1.51565424633, # B
               -0.01025439 * mean_phase + 0.19890163, # C
               0.02121966 * mean_phase + 0.10946198, # D
               0.405218424073, # E
               -0.202987511611 # F
               ]        
        plot_one(disc_data, disc_phase, disc_incidence, disc_emission)

        di_incidence = np.arange(0, 180., 0.1) * oops.RPD
        di_zeros = np.zeros(di_incidence.shape)

        for e in [0, 45, 60]:        
            phot = phot_func(ret, di_incidence, di_zeros+e*oops.RPD)        
            color = colorsys.hsv_to_rgb(np.cos(e*oops.RPD)/2+.5, 1, 1)
            plt.plot(di_incidence*oops.DPR, phot, '-', color=color, lw=1)

        plt.title(filename)
        plt.show()

        plot_one(disc_data, disc_phase, disc_incidence, disc_emission,
                 phot_func=lambda i,e: phot_func(ret, i, e))
        plt.title(filename)
        plt.show()

#===============================================================================
# 
#===============================================================================

def plot_double_minnaert():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    incidence = np.arange(0, 150., 0.1) * oops.RPD
    zeros = np.zeros(incidence.shape)
    
    args = (oops.PI*.75, 
            np.arcsin(0.25*2-1), 0., 0., 0.,
            0., np.arcsin(2.7*oops.PI/oops.TWOPI-1), 0.,
            0.)
    
    new_incidence, new_emission, new_data, phot0 = double_minnaert_func(
           args, incidence, zeros, zeros)
    new_incidence, new_emission, new_data, phot45 = double_minnaert_func(
           args, incidence, zeros+oops.PI/4, zeros)
        
    plt.plot(incidence*oops.DPR, phot0, '-', color='red')
    plt.plot(incidence*oops.DPR, phot45, '-', color='blue')
    plt.show()
    
def test_minnaert_offset():    
    assert False # Convert to full?    
    for full_filename in IMAGE_LIST:
        if type(full_filename) == type((0,)):
            offset_metadata = file_read_offset_metadata(full_filename[0])
            clean_name = file_clean_name(full_filename[0])
        else:
            offset_metadata = file_read_offset_metadata(full_filename)
            clean_name = file_clean_name(full_filename)

        if offset_metadata is None:
            continue

        if offset_metadata['stars_metadata']['offset'] is None:
            continue
            
        for filename in DISC_DATA:
            if filename.endswith(clean_name):
                break
            
        print 'Testing offset for', filename

        full_filename = os.path.join(COISS_2XXX_DERIVED_ROOT, full_filename)
        obs = read_iss_file(full_filename)

        disc_data = DISC_DATA[filename]
        disc_phase = DISC_PHASE[filename]
        disc_incidence = DISC_INCIDENCE[filename]
        disc_emission = DISC_EMISSION[filename]
        
        incidence = disc_incidence
        emission = disc_emission
        phase = disc_phase
        data = disc_data
        
        cutoff = 80 * oops.RPD
        
        data = data[np.where(incidence < cutoff)]
        emission = emission[np.where(incidence < cutoff)]
        phase = phase[np.where(incidence < cutoff)]
        incidence = incidence[np.where(incidence < cutoff)]
        mean_phase = np.mean(full_phase) * oops.DPR
        print 'Mean phase', mean_phase

        if mean_phase > 108:
            continue
                    
        ret = sciopt.fmin_powell(minnaert_opt_func, (0.5,0.5),
                                 args=(incidence, emission, data),
                                 disp=0)
        
        print ret

        k1 = ret[0]
        k2 = ret[1]
        
        new_offset_metadata = master_find_offset(obs, allow_stars=False,
                                                 create_overlay=True)
        
        print 'STAR OFFSET', offset_metadata['stars_metadata']['offset']
        print 'MODEL OFFSET', new_offset_metadata['model_offset']
        
#        new_offset_metadata['offset'] = new_offset_metadata['model_offset']
#        new_offset_metadata['offset'] = offset_metadata['stars_metadata']['offset']
        display_offset_data(obs, new_offset_metadata, show_rings=False)


#==============================================================================
#
# INTERPOLATION FUNCTIONS
# 
#==============================================================================
 
def test_interpolation(phase1, phase2, phase3, filter):
    actual_phase1 = None
    actual_phase2 = None
    actual_phase3 = None
     
    for phase in sorted(DATA_BY_PHASE):
        print phase*oops.DPR
        if abs(phase-phase1*oops.RPD) < 1*oops.RPD:
            actual_phase1 = phase
        if abs(phase-phase2*oops.RPD) < 1*oops.RPD:
            actual_phase2 = phase
        if abs(phase-phase3*oops.RPD) < 1*oops.RPD:
            actual_phase3 = phase
            
    (clean_name1,
     disc_data1,
     disc_phase1,
     disc_incidence1,
     disc_emission1,
     full_data1,
     full_phase1,
     full_incidence1,
     full_emission1) = DATA_BY_PHASE[actual_phase1][filter]

    (clean_name2,
     disc_data2,
     disc_phase2,
     disc_incidence2,
     disc_emission2,
     full_data2,
     full_phase2,
     full_incidence2,
     full_emission2) = DATA_BY_PHASE[actual_phase2][filter]

    (clean_name3,
     disc_data3,
     disc_phase3,
     disc_incidence3,
     disc_emission3,
     full_data3,
     full_phase3,
     full_incidence3,
     full_emission3) = DATA_BY_PHASE[actual_phase3][filter]
    
    points = np.zeros((full_data1.shape[0]+full_data3.shape[0], 3))
    points[:full_data1.shape[0],0] = full_phase1
    points[:full_data1.shape[0],1] = full_incidence1
    points[:full_data1.shape[0],2] = np.cos(full_emission1)
    points[full_data1.shape[0]:,0] = full_phase3
    points[full_data1.shape[0]:,1] = full_incidence3
    points[full_data1.shape[0]:,2] = np.cos(full_emission3)

    points2 = np.zeros((full_data2.shape[0], 3))
    points2[:,0] = full_phase2
    points2[:,1] = full_incidence2
    points2[:,2] = np.cos(full_emission2)
    
    values = np.append(full_data1, full_data3)
    
    new_full_data2 = sciinterp.griddata(points, values, points2, fill_value=0.)
  
    delta = full_data2 - new_full_data2
    print 'Phases', actual_phase1*oops.DPR, actual_phase2*oops.DPR, actual_phase3*oops.DPR
    print 'Mean', np.mean(delta)
    print 'Std', np.std(delta)
    print 'Min', np.min(delta)
    print 'Max', np.max(delta)
    
    full_decimate2 = max(1, len(full_data2) // 10000)
                    
    full_data2 = full_data2[::full_decimate2]
    full_phase2 = full_phase2[::full_decimate2]
    full_incidence2 = full_incidence2[::full_decimate2]
    full_emission2 = full_emission2[::full_decimate2]

    new_full_data2 = new_full_data2[::full_decimate2]

    base_color = (1.,0,0)

    colors = []
    for data_idx in xrange(len(full_data2)):
        shade = np.cos(full_emission2[data_idx])
        color = (base_color[0] * shade,
                 base_color[1] * shade,
                 base_color[2] * shade)
        colors.append(color)
    plt.scatter(full_incidence2*oops.DPR,
                full_data2, s=20, c=colors, alpha=0.05,
                edgecolors='none')

    base_color = (0,0,1.)

    colors = []
    for data_idx in xrange(len(new_full_data2)):
        shade = np.cos(full_emission2[data_idx])
        color = (base_color[0] * shade,
                 base_color[1] * shade,
                 base_color[2] * shade)
        colors.append(color)
    plt.scatter(full_incidence2*oops.DPR,
                new_full_data2, s=20, c=colors, alpha=0.05,
                edgecolors='none')
                        
    plt.xlabel('Incidence angle')
    plt.ylabel('I/F')
    plt.title('PHASE %7.2f' % (actual_phase2*oops.DPR))
    plt.show()


#==============================================================================
#
# MAIN ROUTINES
# 
#==============================================================================

root = tk.Tk()
root.withdraw()

# Make a new big Titan
   
titan = oops.Body.lookup('TITAN')
new_titan = copy.copy(titan)
oops.Body.BODY_REGISTRY['TITAN'] = new_titan

new_titan.radius += 350
new_titan.inner_radius += 350
surface = new_titan.surface
new_titan.surface = oops.surface.Spheroid(surface.origin, surface.frame, (new_titan.radius, new_titan.radius))

titan.name = 'TITAN+ATMOSPHERE'
titan.radius += 650
titan.inner_radius += 650
surface = titan.surface
titan.surface = oops.surface.Spheroid(surface.origin, surface.frame, (titan.radius, titan.radius))
oops.Body.BODY_REGISTRY['TITAN+ATMOSPHERE'] = titan

# 16 - VIO
#find_offset_one_image('COISS_2032/data/1560496833_1560553326/W1560496994_1_CALIB.IMG')
#find_offset_one_image('COISS_2068/data/1680805782_1681997642/W1681926856_1_CALIB.IMG')
#find_offset_one_image('COISS_2068/data/1680805782_1681997642/W1681936403_1_CALIB.IMG')
#find_offset_one_image('COISS_2068/data/1680805782_1681997642/W1681945950_1_CALIB.IMG')

#backplane_one_image('COISS_2032/data/1560496833_1560553326/W1560496994_1_CALIB.IMG', display=True)#, offset=(-1,-4), display=True)#offset=(-1,-2))#display=True, save=False)
#backplane_one_image('COISS_2068/data/1680805782_1681997642/W1681926856_1_CALIB.IMG', offset=(1,-3))#display=True, save=False)
#backplane_one_image('COISS_2068/data/1680805782_1681997642/W1681936403_1_CALIB.IMG', offset=(1,-2))#display=True, save=False)
#backplane_one_image('COISS_2068/data/1680805782_1681997642/W1681945950_1_CALIB.IMG', offset=(0,-1))#display=True, save=False)

# 16 - RED
find_offset_one_image('COISS_2057/data/1629783492_1630072138/W1629929515_1_CALIB.IMG')
#find_offset_one_image('COISS_2068/data/1680805782_1681997642/W1681910412_1_CALIB.IMG')
#find_offset_one_image('COISS_2068/data/1680805782_1681997642/W1681916843_1_CALIB.IMG')
#find_offset_one_image('COISS_2068/data/1680805782_1681997642/W1681926406_1_CALIB.IMG')
#find_offset_one_image('COISS_2068/data/1680805782_1681997642/W1681935953_1_CALIB.IMG')
#find_offset_one_image('COISS_2068/data/1680805782_1681997642/W1681945500_1_CALIB.IMG')

# 166 - VIO
#find_offset_one_image('COISS_2033/data/1561668355_1561837358/W1561794086_1_CALIB.IMG')

#backplane_one_image('COISS_2057/data/1629783492_1630072138/W1629929515_1_CALIB.IMG', offset=(-3,-4), display=True)
#backplane_one_image('COISS_2068/data/1680805782_1681997642/W1681910412_1_CALIB.IMG', offset=())
#backplane_one_image('COISS_2068/data/1680805782_1681997642/W1681916843_1_CALIB.IMG')
#backplane_one_image('COISS_2068/data/1680805782_1681997642/W1681926406_1_CALIB.IMG')
#backplane_one_image('COISS_2068/data/1680805782_1681997642/W1681935953_1_CALIB.IMG')
#backplane_one_image('COISS_2068/data/1680805782_1681997642/W1681945500_1_CALIB.IMG')


# Reprocess all the data to make new backplanes
#backplane_all_images(force=False)

# Read the backplanes
read_backplanes()

#test_interpolation(13, 18, 26, 'VIO')
#test_interpolation(140, 145, 149, 'BL1')

show_plots_multiple_phases(phases=[16.], filters=['RED'])

#show_plots_multiple(phases=[13,18,26], filters=['VIO'])#['RED','GRN', 'BL1', 'VIO'])
#show_plots_multiple(phases=[140,145,149], filters=['BL1'])#['RED','GRN', 'BL1', 'VIO'])
#show_plots_multiple(['CLEAR'])


#===============================================================================
# OLD STUFF 
#===============================================================================

#plot_canned_result()

#plot_params('dm2')

#optimize_disc('arctan1')
#plot_double_minnaert()
#optimize_disc_minnaert()

#optimize_disc_double_minnaert1(name=' 28.65_W1557729551_1')
#optimize_disc_double_minnaert2()

#optimize_disc(alg='dm2', name=' 81.62_W1589361573_1')
#optimize_disc_double_minnaert(name=' 64.59_W1539146659_1')
  
#optimize_disc_double_minnaert2(name=' 71.46_W1540513147_1')

#test_minnaert_offset()
