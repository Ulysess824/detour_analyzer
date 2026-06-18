import sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, ".")
from src.features.feature_engine_features import TSFreshFeatureBuilder, FreshFeatureSelector
from tsfresh import select_features

rng = np.random.default_rng(0)
fechas = [date(2026,1,1)+timedelta(days=i) for i in range(70)]
frames=[]
for planta,sku,base in [("PlantaA","SKU1",50.0),("PlantaB","SKU2",12.0)]:
    s=np.empty(70); s[0]=base
    for t in range(1,70): s[t]=max(0.0,0.9*s[t-1]+rng.normal(0,base*0.05)+base*0.1)
    frames.append(pl.DataFrame({"planta":planta,"sku":sku,"fecha":fechas,"consumo_real":s,"forecast_mensual":base*20.0}).with_columns(target_consumo_real=pl.col("consumo_real").shift(-1).over(["planta","sku"])))
panel=pl.concat(frames).sort(["planta","sku","fecha"])
train=panel.filter(pl.col("fecha")<=pl.date(2026,2,19))

b=TSFreshFeatureBuilder(fc_parameters="efficient",max_timeshift=10,min_timeshift=3)
feat,cols=b.fit_transform(train)
sel=FreshFeatureSelector(fdr_level=0.05).fit(feat,cols,y="target_consumo_real")
mine=set(sel.selected_features_)

# tsfresh reference: same X/y, drop null target/feature rows, impute
import pandas as pd
pdf=feat.to_pandas()
y=pdf["target_consumo_real"]
valid=y.notna()
X=pdf.loc[valid,cols].fillna(0.0)
yv=y.loc[valid]
ref=set(select_features(X,yv,fdr_level=0.05).columns)

print("mine:",len(mine),"tsfresh:",len(ref))
print("intersection:",len(mine&ref))
print("jaccard:",round(len(mine&ref)/len(mine|ref),3))
print("only mine:",len(mine-ref),"only tsfresh:",len(ref-mine))
