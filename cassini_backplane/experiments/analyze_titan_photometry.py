import colorsys
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sciopt

import oops

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
    # 13
    ('COISS_2057/data/1629783492_1630072138/W1629929416_1_CALIB.IMG', # VIO
     'COISS_2057/data/1629783492_1630072138/W1629929449_1_CALIB.IMG', # BL1
     'COISS_2057/data/1629783492_1630072138/W1629929482_1_CALIB.IMG', # GRN
     'COISS_2057/data/1629783492_1630072138/W1629929515_1_CALIB.IMG', # RED
     'COISS_2007/data/1477438740_1477481438/W1477471954_6_CALIB.IMG', # IR3
     'COISS_2057/data/1629783492_1630072138/W1629930385_1_CALIB.IMG', # CB2
     'COISS_2057/data/1629783492_1630072138/W1629929871_1_CALIB.IMG', # CB3
     'COISS_2057/data/1629783492_1630072138/W1629930219_1_CALIB.IMG', # MT2
     'COISS_2057/data/1629783492_1630072138/W1629929965_1_CALIB.IMG', # MT3
    ),
    
    # 18               
    ('COISS_2068/data/1680805782_1681997642/W1681908315_1_CALIB.IMG', # VIO
     'COISS_2068/data/1680805782_1681997642/W1681910478_1_CALIB.IMG', # BL1
     'COISS_2068/data/1680805782_1681997642/W1681908381_1_CALIB.IMG', # GRN
     'COISS_2068/data/1680805782_1681997642/W1681908414_1_CALIB.IMG', # RED
     'COISS_2068/data/1680805782_1681997642/W1681909321_1_CALIB.IMG', # CB2
     'COISS_2068/data/1680805782_1681997642/W1681909505_1_CALIB.IMG', # CB3
     'COISS_2068/data/1680805782_1681997642/W1681909361_1_CALIB.IMG', # MT2
     'COISS_2068/data/1680805782_1681997642/W1681909433_1_CALIB.IMG', # MT3
    ),
                       
    # 23
    ('COISS_2016/data/1508882988_1509138645/W1509136363_2_CALIB.IMG', # VIO
     'COISS_2016/data/1508882988_1509138645/W1509136315_2_CALIB.IMG', # BL1
     'COISS_2016/data/1508882988_1509138645/W1509136282_2_CALIB.IMG', # GRN
     'COISS_2016/data/1508882988_1509138645/W1509136249_2_CALIB.IMG', # RED
     'COISS_2016/data/1508882988_1509138645/W1509138181_2_CALIB.IMG', # IR1
     'COISS_2016/data/1508882988_1509138645/W1509136446_2_CALIB.IMG', # CB2
     'COISS_2016/data/1508882988_1509138645/W1509137510_2_CALIB.IMG', # CB3
     'COISS_2016/data/1508882988_1509138645/W1509136486_2_CALIB.IMG', # MT2
     'COISS_2016/data/1508882988_1509138645/W1509136099_2_CALIB.IMG', # MT3 
    ),
                       
    # 28
    ('COISS_2017/data/1514219847_1514299401/W1514281726_1_CALIB.IMG', # VIO
     'COISS_2017/data/1514219847_1514299401/W1514281743_1_CALIB.IMG', # BL1
     'COISS_2017/data/1514219847_1514299401/W1514281760_1_CALIB.IMG', # GRN
     'COISS_2017/data/1514219847_1514299401/W1514281777_1_CALIB.IMG', # RED
     'COISS_2017/data/1514219847_1514299401/W1514284221_1_CALIB.IMG', # IR1
     'COISS_2017/data/1514219847_1514299401/W1514281650_1_CALIB.IMG', # CB3
     'COISS_2017/data/1514219847_1514299401/W1514284552_1_CALIB.IMG', # MT3
     ),
                       
    # 32
    ('COISS_2063/data/1655742537_1655905033/W1655808265_1_CALIB.IMG', # VIO
     'COISS_2063/data/1655742537_1655905033/W1655808298_1_CALIB.IMG', # BL1
     'COISS_2063/data/1655742537_1655905033/W1655808331_1_CALIB.IMG', # GRN
     'COISS_2063/data/1655742537_1655905033/W1655808364_1_CALIB.IMG', # RED
     'COISS_2063/data/1655742537_1655905033/W1655809059_1_CALIB.IMG', # CB2
     'COISS_2063/data/1655742537_1655905033/W1655808656_1_CALIB.IMG', # CB3
     'COISS_2063/data/1655742537_1655905033/W1655809004_1_CALIB.IMG', # MT2
     'COISS_2063/data/1655742537_1655905033/W1655808750_1_CALIB.IMG', # MT3
     ),
                       
    # 36
    ('COISS_2055/data/1624240547_1624420949/W1624420234_1_CALIB.IMG', # VIO
     'COISS_2055/data/1624240547_1624420949/W1624420267_1_CALIB.IMG', # BL1
     'COISS_2055/data/1624240547_1624420949/W1624420300_1_CALIB.IMG', # GRN
     'COISS_2055/data/1624240547_1624420949/W1624420333_1_CALIB.IMG', # RED
     'COISS_2055/data/1624240547_1624420949/W1624419266_1_CALIB.IMG', # CB2
     'COISS_2055/data/1624240547_1624420949/W1624420657_1_CALIB.IMG', # CB3
     'COISS_2055/data/1624240547_1624420949/W1624419283_1_CALIB.IMG', # MT2
     'COISS_2055/data/1624240547_1624420949/W1624420751_1_CALIB.IMG', # MT3
    ),
    
    # 46
    ('COISS_2068/data/1683372651_1683615321/W1683615178_1_CALIB.IMG', # VIO
     'COISS_2068/data/1683372651_1683615321/W1683615211_1_CALIB.IMG', # BL1
     'COISS_2068/data/1683372651_1683615321/W1683615266_1_CALIB.IMG', # GRN
     'COISS_2068/data/1683372651_1683615321/W1683615321_1_CALIB.IMG', # RED
     'COISS_2068/data/1683616251_1683727987/W1683616251_1_CALIB.IMG', # CB2
     'COISS_2068/data/1683616251_1683727987/W1683616447_1_CALIB.IMG', # CB3
     'COISS_2068/data/1683616251_1683727987/W1683616313_1_CALIB.IMG', # MT2
     'COISS_2068/data/1683616251_1683727987/W1683616391_1_CALIB.IMG', # MT3
     ),
                       
    # 55
    ('COISS_2011/data/1492217706_1492344437/W1492341257_2_CALIB.IMG', # VIO
     'COISS_2011/data/1492217706_1492344437/W1492341209_2_CALIB.IMG', # BL1
     'COISS_2011/data/1492217706_1492344437/W1492341176_2_CALIB.IMG', # GRN
     'COISS_2011/data/1492217706_1492344437/W1492341143_2_CALIB.IMG', # RED
     'COISS_2011/data/1492217706_1492344437/W1492341292_2_CALIB.IMG', # CB2
     'COISS_2011/data/1492217706_1492344437/W1492342134_2_CALIB.IMG', # CB3
     'COISS_2011/data/1492217706_1492344437/W1492341332_2_CALIB.IMG', # MT2
     'COISS_2011/data/1492217706_1492344437/W1492341666_2_CALIB.IMG', # MT3
    ),
                       
    # 60
    ('COISS_2030/data/1552197101_1552225837/W1552216646_1_CALIB.IMG', # VIO
     'COISS_2030/data/1552197101_1552225837/W1552216606_1_CALIB.IMG', # BL1
     'COISS_2030/data/1552197101_1552225837/W1552216573_1_CALIB.IMG', # GRN
     'COISS_2030/data/1552197101_1552225837/W1552216540_1_CALIB.IMG', # RED
     'COISS_2030/data/1552197101_1552225837/W1552216381_1_CALIB.IMG', # CB2
     'COISS_2030/data/1552197101_1552225837/W1552216035_1_CALIB.IMG', # CB3
     'COISS_2030/data/1552197101_1552225837/W1552216451_1_CALIB.IMG', # MT2
     'COISS_2030/data/1552197101_1552225837/W1552216219_1_CALIB.IMG', # MT3
    ),
                       
    # 68
    ('COISS_2030/data/1550711108_1550921975/W1550838842_1_CALIB.IMG', # VIO
     'COISS_2030/data/1550711108_1550921975/W1550838802_1_CALIB.IMG', # BL1
     'COISS_2030/data/1550711108_1550921975/W1550838769_1_CALIB.IMG', # GRN
     'COISS_2030/data/1550711108_1550921975/W1550838736_1_CALIB.IMG', # RED
     'COISS_2030/data/1550711108_1550921975/W1550838572_1_CALIB.IMG', # CB2
     'COISS_2030/data/1550711108_1550921975/W1550838210_1_CALIB.IMG', # CB3
     'COISS_2030/data/1550711108_1550921975/W1550838647_1_CALIB.IMG', # MT2
     'COISS_2030/data/1550711108_1550921975/W1550838394_1_CALIB.IMG', # MT3
    ),
                       
    # 73
    ('COISS_2082/data/1743902905_1744323160/W1743914297_1_CALIB.IMG', # VIO
     'COISS_2082/data/1743902905_1744323160/W1743914227_1_CALIB.IMG', # BL1
     'COISS_2082/data/1743902905_1744323160/W1743914172_1_CALIB.IMG', # GRN
     'COISS_2082/data/1743902905_1744323160/W1743914117_1_CALIB.IMG', # RED
    ),
                       
    # 80
    ('COISS_2082/data/1748047605_1748159948/W1748064618_1_CALIB.IMG', # CLEAR
     'COISS_2082/data/1748047605_1748159948/W1748065141_1_CALIB.IMG', # VIO
     'COISS_2082/data/1748047605_1748159948/W1748065093_3_CALIB.IMG', # BL1
     'COISS_2082/data/1748047605_1748159948/W1748065060_1_CALIB.IMG', # GRN
     'COISS_2082/data/1748047605_1748159948/W1748065027_1_CALIB.IMG', # RED
     'COISS_2082/data/1748047605_1748159948/W1748064784_1_CALIB.IMG', # CB3
     'COISS_2082/data/1748047605_1748159948/W1748064690_3_CALIB.IMG', # MT3
    ),
                       
    # 85
    ('COISS_2085/data/1757258657_1757635754/W1757630543_1_CALIB.IMG', # BL1
     'COISS_2085/data/1757258657_1757635754/W1757630576_1_CALIB.IMG', # GRN
     'COISS_2085/data/1757258657_1757635754/W1757630609_1_CALIB.IMG', # RED
     'COISS_2085/data/1757258657_1757635754/W1757630775_1_CALIB.IMG', # CB3
    ),
                       
    # 90
    ('COISS_2084/data/1753489519_1753577899/W1753508727_1_CALIB.IMG', # VIO
     'COISS_2084/data/1753489519_1753577899/W1753508760_1_CALIB.IMG', # BL1
     'COISS_2084/data/1753489519_1753577899/W1753508793_1_CALIB.IMG', # GRN
     'COISS_2084/data/1753489519_1753577899/W1753508826_1_CALIB.IMG', # RED
     'COISS_2084/data/1753489519_1753577899/W1753507284_1_CALIB.IMG', # CB2
     'COISS_2084/data/1753489519_1753577899/W1753508679_1_CALIB.IMG', # CB3
     'COISS_2084/data/1753489519_1753577899/W1753507324_1_CALIB.IMG', # MT2
     'COISS_2084/data/1753489519_1753577899/W1753508607_1_CALIB.IMG', # MT3
    ),
    
    # 95
    ('COISS_2049/data/1604402501_1604469049/W1604460899_1_CALIB.IMG', # VIO
     'COISS_2049/data/1604402501_1604469049/W1604460932_1_CALIB.IMG', # BL1
     'COISS_2049/data/1604402501_1604469049/W1604460965_1_CALIB.IMG', # GRN
     'COISS_2049/data/1604402501_1604469049/W1604460998_1_CALIB.IMG', # RED
     'COISS_2049/data/1604402501_1604469049/W1604461827_1_CALIB.IMG', # CB2
     'COISS_2049/data/1604402501_1604469049/W1604461370_1_CALIB.IMG', # CB3
     'COISS_2049/data/1604402501_1604469049/W1604461794_1_CALIB.IMG', # MT2
     'COISS_2049/data/1604402501_1604469049/W1604461464_1_CALIB.IMG', # MT3
    ),
                       
    # 104
    ('COISS_2028/data/1548522106_1548756214/W1548712789_1_CALIB.IMG', # VIO
     'COISS_2028/data/1548522106_1548756214/W1548712741_1_CALIB.IMG', # BL1
     'COISS_2028/data/1548522106_1548756214/W1548712708_1_CALIB.IMG', # GRN
     'COISS_2028/data/1548522106_1548756214/W1548712675_1_CALIB.IMG', # RED
     'COISS_2028/data/1548522106_1548756214/W1548716874_1_CALIB.IMG', # IR1
     'COISS_2028/data/1548522106_1548756214/W1548712208_1_CALIB.IMG', # CB3
    ),
    
    # 110
    ('COISS_2086/data/1760272897_1760532147/W1760465018_1_CALIB.IMG', # VIO
     'COISS_2086/data/1760272897_1760532147/W1760465051_1_CALIB.IMG', # BL1
     'COISS_2086/data/1760272897_1760532147/W1760464970_1_CALIB.IMG', # RED
     'COISS_2086/data/1760272897_1760532147/W1760464915_1_CALIB.IMG', # CB3
    ),
                       
    # 114
    ('COISS_2030/data/1552128674_1552196996/W1552152072_1_CALIB.IMG', # VIO
     'COISS_2030/data/1552128674_1552196996/W1552152032_1_CALIB.IMG', # BL1
     'COISS_2030/data/1552128674_1552196996/W1552151999_1_CALIB.IMG', # GRN
     'COISS_2030/data/1552128674_1552196996/W1552151966_1_CALIB.IMG', # RED
     'COISS_2030/data/1552128674_1552196996/W1552151900_1_CALIB.IMG', # CB2
     'COISS_2030/data/1552128674_1552196996/W1552151819_1_CALIB.IMG', # CB3
     'COISS_2030/data/1552128674_1552196996/W1552151933_1_CALIB.IMG', # MT2
     'COISS_2030/data/1552128674_1552196996/W1552151867_1_CALIB.IMG', # MT3
    ),
    
    # 119
    ('COISS_2024/data/1532184650_1532257621/W1532185013_1_CALIB.IMG', # VIO
     'COISS_2024/data/1532184650_1532257621/W1532185046_1_CALIB.IMG', # BL1
     'COISS_2024/data/1532184650_1532257621/W1532184980_1_CALIB.IMG', # GRN
     'COISS_2024/data/1532184650_1532257621/W1532184947_1_CALIB.IMG', # RED
     'COISS_2024/data/1532136934_1532184428/W1532183802_1_CALIB.IMG', # IR1
     'COISS_2024/data/1532184650_1532257621/W1532185079_1_CALIB.IMG', # IR3
     'COISS_2024/data/1532184650_1532257621/W1532184650_1_CALIB.IMG', # IR5
     'COISS_2024/data/1532184650_1532257621/W1532184723_1_CALIB.IMG', # CB2
     'COISS_2024/data/1532136934_1532184428/W1532184214_1_CALIB.IMG', # MT2
     'COISS_2024/data/1532136934_1532184428/W1532183891_1_CALIB.IMG', # MT3
    ),
                       
    # 125
    ('COISS_2015/data/1503416732_1503874914/W1503421986_2_CALIB.IMG', # VIO
     'COISS_2015/data/1503416732_1503874914/W1503422118_2_CALIB.IMG', # BL1
     'COISS_2015/data/1503416732_1503874914/W1503422052_2_CALIB.IMG', # GRN
     'COISS_2015/data/1503416732_1503874914/W1503422085_2_CALIB.IMG', # RED
    ),
                       
    # 130
    ('COISS_2055/data/1622711732_1623166344/W1622977122_1_CALIB.IMG', # VIO
     'COISS_2055/data/1622711732_1623166344/W1622977155_1_CALIB.IMG', # BL1
     'COISS_2055/data/1622711732_1623166344/W1622977188_1_CALIB.IMG', # GRN
     'COISS_2055/data/1622711732_1623166344/W1622977221_1_CALIB.IMG', # RED
     'COISS_2055/data/1622711732_1623166344/W1622977738_1_CALIB.IMG', # CB2
     'COISS_2055/data/1622711732_1623166344/W1622977497_1_CALIB.IMG', # CB3
     'COISS_2055/data/1622711732_1623166344/W1622977705_1_CALIB.IMG', # MT2
     'COISS_2055/data/1622711732_1623166344/W1622977567_1_CALIB.IMG', # MT3
    ),
                       
    # 135
    ('COISS_2060/data/1643317802_1643406946/W1643375992_1_CALIB.IMG', # VIO
     'COISS_2060/data/1643317802_1643406946/W1643376025_1_CALIB.IMG', # BL1
     'COISS_2060/data/1643317802_1643406946/W1643376058_1_CALIB.IMG', # GRN
     'COISS_2060/data/1643317802_1643406946/W1643376091_1_CALIB.IMG', # RED
     'COISS_2060/data/1643317802_1643406946/W1643376539_1_CALIB.IMG', # CB2
     'COISS_2060/data/1643317802_1643406946/W1643376327_1_CALIB.IMG', # CB3
     'COISS_2060/data/1643317802_1643406946/W1643376506_1_CALIB.IMG', # MT2
     'COISS_2060/data/1643317802_1643406946/W1643376375_1_CALIB.IMG', # MT3
    ),
                       
    # 140
    ('COISS_2076/data/1727255245_1727449737/W1727330741_1_CALIB.IMG', # VIO
     'COISS_2076/data/1727255245_1727449737/W1727330774_1_CALIB.IMG', # BL1
     'COISS_2076/data/1727255245_1727449737/W1727330807_1_CALIB.IMG', # GRN
     'COISS_2076/data/1727255245_1727449737/W1727330840_1_CALIB.IMG', # RED
     'COISS_2076/data/1727255245_1727449737/W1727332645_1_CALIB.IMG', # CB2
     'COISS_2076/data/1727255245_1727449737/W1727332801_1_CALIB.IMG', # CB3
     'COISS_2076/data/1727255245_1727449737/W1727332572_1_CALIB.IMG', # MT2
     'COISS_2076/data/1727255245_1727449737/W1727332734_1_CALIB.IMG', # MT3
    ),
                       
    # 145
    ('COISS_2062/data/1652952601_1653081456/W1652985001_1_CALIB.IMG', # VIO
     'COISS_2062/data/1652952601_1653081456/W1652985034_1_CALIB.IMG', # BL1
     'COISS_2062/data/1652952601_1653081456/W1652985067_1_CALIB.IMG', # GRN
     'COISS_2062/data/1652952601_1653081456/W1652985100_1_CALIB.IMG', # RED
     'COISS_2062/data/1652952601_1653081456/W1652985344_1_CALIB.IMG', # CB3
     'COISS_2062/data/1652952601_1653081456/W1652985414_1_CALIB.IMG', # MT3
    ),
                       
    # 149
    ('COISS_2076/data/1721802517_1721894741/W1721822901_1_CALIB.IMG', # CLEAR
     'COISS_2076/data/1721802517_1721894741/W1721822934_1_CALIB.IMG', # VIO
     'COISS_2076/data/1721802517_1721894741/W1721822967_1_CALIB.IMG', # BL1
     'COISS_2076/data/1721802517_1721894741/W1721823000_1_CALIB.IMG', # GRN
     'COISS_2076/data/1721802517_1721894741/W1721823033_1_CALIB.IMG', # RED
     'COISS_2076/data/1721802517_1721894741/W1721823066_1_CALIB.IMG', # IR1
     'COISS_2076/data/1721802517_1721894741/W1721823382_1_CALIB.IMG', # IR2
     'COISS_2076/data/1721802517_1721894741/W1721823099_1_CALIB.IMG', # IR3
     'COISS_2076/data/1721802517_1721894741/W1721823316_1_CALIB.IMG', # CB2
     'COISS_2076/data/1721802517_1721894741/W1721823147_1_CALIB.IMG', # CB3
     'COISS_2076/data/1721802517_1721894741/W1721823349_1_CALIB.IMG', # MT2
     'COISS_2076/data/1721802517_1721894741/W1721823283_1_CALIB.IMG', # MT3
    ),
                       
    # 155
    ('COISS_2074/data/1717673485_1717756625/W1717682790_1_CALIB.IMG', # VIO
     'COISS_2074/data/1717673485_1717756625/W1717688223_1_CALIB.IMG', # BL1
     'COISS_2074/data/1717673485_1717756625/W1717682856_1_CALIB.IMG', # GRN
     'COISS_2074/data/1717673485_1717756625/W1717688289_1_CALIB.IMG', # RED
     'COISS_2074/data/1717673485_1717756625/W1717688370_1_CALIB.IMG', # MT3
    ),
                       
    # 158
    ('COISS_2009/data/1487140530_1487182149/W1487166992_2_CALIB.IMG', # CLEAR
     'COISS_2009/data/1487140530_1487182149/W1487167364_2_CALIB.IMG', # VIO
     'COISS_2009/data/1487140530_1487182149/W1487167243_2_CALIB.IMG', # BL1
     'COISS_2009/data/1487140530_1487182149/W1487167154_2_CALIB.IMG', # GRN
     'COISS_2009/data/1487140530_1487182149/W1487167033_2_CALIB.IMG', # RED
    ),
    
    # 164
    ('COISS_2074/data/1716194058_1716328931/W1716307950_1_CALIB.IMG', # VIO
     'COISS_2074/data/1716194058_1716328931/W1716307983_1_CALIB.IMG', # BL1
     'COISS_2074/data/1716194058_1716328931/W1716308016_1_CALIB.IMG', # GRN
     'COISS_2074/data/1716194058_1716328931/W1716308049_1_CALIB.IMG', # RED
    ),
                       
    # 166
    ('COISS_2033/data/1561668355_1561837358/W1561790952_1_CALIB.IMG', # CLEAR
     'COISS_2033/data/1561668355_1561837358/W1561794086_1_CALIB.IMG', # VIO
     'COISS_2033/data/1561668355_1561837358/W1561794119_1_CALIB.IMG', # BL1
     'COISS_2033/data/1561668355_1561837358/W1561794152_1_CALIB.IMG', # GRN
     'COISS_2033/data/1561668355_1561837358/W1561794185_1_CALIB.IMG', # RED
     'COISS_2033/data/1561668355_1561837358/W1561794349_1_CALIB.IMG', # MT2
    ),
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

def backplane_one_image(filename_list):
    print 'Analyzing', filename_list
    if type(filename_list) != type((0,)):
        filename_list = [filename_list]

    disc_phase = np.array([])
    disc_incidence = np.array([])
    disc_emission = np.array([])
    disc_data = np.array([])
    atmos_phase = np.array([])
    atmos_incidence = np.array([])
    atmos_emission = np.array([])
    atmos_data = np.array([])
    
    for filename in filename_list:
        full_filename = os.path.join(COISS_2XXX_DERIVED_ROOT, filename)
        obs = read_iss_file(full_filename)
        offset_metadata = file_read_offset_metadata(full_filename)
    
        if offset_metadata is None:
            print 'Warning: No offset data for', filename
        else:
            offset = offset_metadata['offset']
            obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
    
        bp = oops.Backplane(obs)
        
        phase = bp.phase_angle('TITAN+ATMOSPHERE')
        incidence = bp.incidence_angle('TITAN+ATMOSPHERE')
        emission = bp.emission_angle('TITAN+ATMOSPHERE')
        phase2 = bp.phase_angle('TITAN')

        full_mask = np.logical_not(phase.mask)
        full_mask[:5,:] = 0
        full_mask[-5:,:] = 0
        full_mask[:,:5] = 0
        full_mask[:,-5:] = 0
        
        disc_mask = np.logical_not(phase2.mask)
        disc_mask[:5,:] = 0
        disc_mask[-5:,:] = 0
        disc_mask[:,:5] = 0
        disc_mask[:,-5:] = 0

        atmos_mask = np.logical_and(full_mask, np.logical_not(disc_mask))
        
        disc_phase     = np.append(disc_phase,
                                     phase.vals[disc_mask].flatten())
        disc_incidence = np.append(disc_incidence, 
                                     incidence.vals[disc_mask].flatten())
        disc_emission  = np.append(disc_emission,
                                     emission.vals[disc_mask].flatten())
        disc_data      = np.append(disc_data,
                                     obs.data[disc_mask].flatten())

        atmos_phase     = np.append(atmos_phase,
                                    phase.vals[atmos_mask].flatten())
        atmos_incidence = np.append(atmos_incidence,
                                    incidence.vals[atmos_mask].flatten())
        atmos_emission  = np.append(atmos_emission,
                                    emission.vals[atmos_mask].flatten())
        atmos_data      = np.append(atmos_data,
                                    obs.data[atmos_mask].flatten())
        
    mean_phase = np.mean(disc_phase) * oops.DPR
    print obs.data.shape, obs.filter1, obs.filter2,
        
    clean_name = file_clean_name(filename_list[0])
    ph_name = ('%6.2f' % mean_phase) + '_' + clean_name

    print 'Mean phase %6.2f' % mean_phase,
    print ph_name

    np.savez(os.path.join(RESULTS_ROOT, 'titan', clean_name),
             disc_phase=disc_phase, disc_incidence=disc_incidence, 
             disc_emission=disc_emission, disc_data=disc_data,
             atmos_phase=atmos_phase, atmos_incidence=atmos_incidence, 
             atmos_emission=atmos_emission, atmos_data=atmos_data)
     
def backplane_all_images(force=False):
    for image_list in IMAGE_LIST_BY_PHASE:
        for filename in image_list:
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

DATA_BY_PHASE = {}

IMAGE_NAME = {}
DISC_DATA = {}
DISC_PHASE = {}
DISC_INCIDENCE = {}
DISC_EMISSION = {}
ATMOS_DATA = {}
ATMOS_PHASE = {}
ATMOS_INCIDENCE = {}
ATMOS_EMISSION = {}

def read_backplanes():
    for image_list in IMAGE_LIST_BY_PHASE:
        image_dict = None
        for filename in image_list:
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
            
            disc_data = np_arrays['disc_data']
            disc_phase = np_arrays['disc_phase']
            disc_incidence = np_arrays['disc_incidence']
            disc_emission = np_arrays['disc_emission']
            atmos_data = np_arrays['atmos_data']
            atmos_phase = np_arrays['atmos_phase']
            atmos_incidence = np_arrays['atmos_incidence']
            atmos_emission = np_arrays['atmos_emission']
            mean_phase = np.mean(disc_phase) * oops.DPR

            if not image_dict:
                mean_phase = np.mean(disc_phase)
                image_dict = {}
                DATA_BY_PHASE[mean_phase] = image_dict

            if obs.filter1 == 'CL1' and obs.filter2 == 'CL2':
                filter = 'CLEAR'
            else:
                filter = obs.filter1
                if filter == 'CL1':
                    filter = obs.filter2
                elif obs.filter2 != 'CL2':
                    filter += '+' + obs.filter2
                    
            image_dict[filter] = (clean_name,
                                  disc_data,
                                  disc_phase,
                                  disc_incidence,
                                  disc_emission,
                                  atmos_data,
                                  atmos_phase,
                                  atmos_incidence,
                                  atmos_emission)
    
def show_plots(name=None, use_minnaert=True):
    for filename in sorted(DISC_DATA):
        print filename, name
        if name is not None and filename != name:
            continue
        disc_data = DISC_DATA[filename]
        disc_phase = DISC_PHASE[filename]
        disc_incidence = DISC_INCIDENCE[filename]
        disc_emission = DISC_EMISSION[filename]
        atmos_data = ATMOS_DATA[filename]
        atmos_phase = ATMOS_PHASE[filename]
        atmos_incidence = ATMOS_INCIDENCE[filename]
        atmos_emission = ATMOS_EMISSION[filename]

        if use_minnaert:
            cutoff = 80 * oops.RPD
            
            odata = disc_data[np.where(disc_incidence < cutoff)]
            oemission = disc_emission[np.where(disc_incidence < cutoff)]
            ophase = disc_phase[np.where(disc_incidence < cutoff)]
            oincidence = disc_incidence[np.where(disc_incidence < cutoff)]
            mean_phase = np.mean(disc_phase) * oops.DPR
            print 'Mean phase', mean_phase
            if mean_phase > 108:
                continue
            
            mi, me = sciopt.fmin(minnaert_opt_func, (0.5,0.5),
                                 args=(oincidence, oemission, odata),
                                 disp=0)

        plot_one(disc_data, disc_phase, disc_incidence, disc_emission)
        plt.title(filename)
        plt.show()
        
def show_plots_multiple(filters=None, phases=None):
    for phase in sorted(DATA_BY_PHASE):
        if phases is not None:
            for phase2 in phases:
                if abs(phase-phase2*oops.RPD) < 1*oops.RPD:
                    break
            else:
                continue

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        used_filters = []
        
        for filter in sorted(DATA_BY_PHASE[phase]):
            if filters and filter not in filters:
                continue
            
            (clean_name,
             disc_data,
             disc_phase,
             disc_incidence,
             disc_emission,
             atmos_data,
             atmos_phase,
             atmos_incidence,
             atmos_emission) = DATA_BY_PHASE[phase][filter]
                
            used_filters.append(filter)

            disc_decimate = 1#max(1, len(disc_data) // 10000)
            atmos_decimate = 100
                    
            disc_data = disc_data[::disc_decimate]
            disc_phase = disc_phase[::disc_decimate]
            disc_incidence = disc_incidence[::disc_decimate]
            disc_emission = disc_emission[::disc_decimate]

            if filter not in FILTER_COLOR:
                print 'Unknown filter', filter, 'file', clean_name
                continue
            
            base_color = FILTER_COLOR[filter]

            colors = []
            for data_idx in xrange(len(disc_data)):
                shade = np.cos(disc_emission[data_idx])
                color = (base_color[0] * shade,
                         base_color[1] * shade,
                         base_color[2] * shade)
                colors.append(color)
            plt.scatter(disc_incidence*oops.DPR,
                        disc_data, s=20, c=colors, alpha=0.05,
                        edgecolors='none')
                        
        xlimits = plt.xlim()
        ylimits = plt.ylim()
        
        for filter in sorted(DATA_BY_PHASE[phase]):
            if filter not in FILTER_COLOR:
                continue
            plt.plot(-100,-100, 'o', mec=FILTER_COLOR[filter], mfc=FILTER_COLOR[filter], ms=10,
                     label=filter)

        plt.xlim(0, xlimits[1])
        plt.ylim(0, ylimits[1])
        
        plt.legend(numpoints=1)
        plt.xlabel('Incidence angle')
        plt.ylabel('I/F')
        plt.title('PHASE %7.2f' % (phase*oops.DPR))
        plt.show()
        

#===============================================================================
#
# PLOTTING HELPERS
# 
#===============================================================================

def plot_one(disc_data, disc_phase, disc_incidence, disc_emission, 
             atmos_data=None, atmos_phase=None, atmos_incidence=None, 
             atmos_emission=None,
             phot_func=None):
    disc_decimate = len(disc_data) // 1000
    atmos_decimate = 100
            
#    disc_decimate = 10
    
    disc_data = disc_data[::disc_decimate]
    disc_phase = disc_phase[::disc_decimate]
    disc_incidence = disc_incidence[::disc_decimate]
    disc_emission = disc_emission[::disc_decimate]
    
    phase_list = [0]
    if atmos_data:
        atmos_data = atmos_data[::atmos_decimate]
        atmos_phase = atmos_phase[::atmos_decimate]
        atmos_incidence = atmos_incidence[::atmos_decimate]
        atmos_emission = atmos_emission[::atmos_decimate]
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
            incidence = atmos_incidence
            emission = atmos_emission
            data = atmos_data
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
        
        mean_phase = np.mean(disc_phase) * oops.DPR
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
    phase_list = []
    ret_list = []
    
    phot_func, opt_func, angle_func, guess = get_funcs(alg)
    
    for filename in sorted(DISC_DATA):
        print 'Optimizing', alg, filename

        disc_data = DISC_DATA[filename]
        disc_phase = DISC_PHASE[filename]
        disc_incidence = DISC_INCIDENCE[filename]
        disc_emission = DISC_EMISSION[filename]

        mean_phase = np.mean(disc_phase) * oops.DPR
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
        
        mean_phase = np.mean(disc_phase) * oops.DPR
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
        mean_phase = np.mean(disc_phase) * oops.DPR
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
 
 
    
    
titan = oops.Body.lookup('TITAN')
new_titan = copy.copy(titan)
oops.Body.BODY_REGISTRY['TITAN'] = new_titan

new_titan.radius += 350
new_titan.inner_radius += 350
surface = new_titan.surface
new_titan.surface = oops.surface.Spheroid(surface.origin, surface.frame, (new_titan.radius, new_titan.radius))

titan.name = 'TITAN+ATMOSPHERE'
titan.radius += 350
titan.inner_radius += 350
surface = titan.surface
titan.surface = oops.surface.Spheroid(surface.origin, surface.frame, (titan.radius, titan.radius))
oops.Body.BODY_REGISTRY['TITAN+ATMOSPHERE'] = titan

#backplane_all_images(force=False)

read_backplanes()

show_plots_multiple(phases=[60])#['RED','GRN', 'BL1', 'VIO'])
#show_plots_multiple(['CLEAR'])

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
