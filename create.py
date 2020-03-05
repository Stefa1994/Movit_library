import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import geopy.distance
from geopy.distance import distance
import time


def foo(x, y):
    return (x,y)

def diff_custom(x,y):
  vals = distance(x,y).m
  return(vals)
  
def cumulative(x,y):
    vals = x + timedelta(days = 0, seconds = round(y))
    return(vals)


def create(fpath):
	raw = pd.read_csv(fpath, 
				index_col = False,
				parse_dates=['start_time',"stop_time"]
			   )
	raw = raw.reset_index().rename(columns={"index":"path_id"})
	raw["trip_duration"] = raw["stop_time"] - raw["start_time"]
	raw["trip_duration"] = raw["trip_duration"].astype('timedelta64[s]')
	df_ridotto = pd.read_csv('path.csv')
	df_ridotto["lat_lon_start"] = df_ridotto.apply(lambda x: foo(x['from_lat'],x['from_lon']), axis = 1)
	df_ridotto["lat_lon_end"] = df_ridotto.apply(lambda x: foo(x['to_lat'],x['to_lon']), axis = 1)
	df_ridotto["distance"] = df_ridotto.apply(lambda x: diff_custom(x['lat_lon_start'], x['lat_lon_end']), axis = 1)
	df_ridotto['distance'].iloc[0] = 0
	df_ridotto['path_id'] = df_ridotto["path_id"] - 1
	unione = pd.merge(df_ridotto,raw,how = "left",on="path_id")
	unione = unione[unione["calculated"]!=False]
	unione = unione.drop(["calculated","lat_lon_start","lat_lon_end"], axis = 1)
	lista_distance = unione.groupby(["path_id"])["distance"].sum()
	unione = pd.merge(unione,lista_distance,how = "left",on="path_id")
	unione = unione.rename(columns={"distance_y":"distance_tot","distance_x":"distance"})
	unione["mean_velocityms"] = (unione["distance_tot"] / unione["trip_duration"])
	unione["transfer_time"] = unione["distance"]/unione["mean_velocityms"]
	unione["cumulative_transfer_time"] = unione.groupby("path_id")["transfer_time"].cumsum()
	unione["time_progress"] = unione.apply(lambda x: cumulative(x["start_time"],x["cumulative_transfer_time"]),axis=1) 

	return unione

