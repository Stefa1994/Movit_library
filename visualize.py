import folium
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, date
from PIL import Image, ImageDraw, ImageFont
import io
import time
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb
import scipy.ndimage.filters
from IPython.display import IFrame
from geopy.geocoders import Nominatim
import os
from datetime import datetime, timedelta

def substitute(x,y):
    if(pd.isnull(x)):
        x = y
    return x

def get_min_max(bike_data):
    min_lat = bike_data["from_lat"].min()
    max_lat = bike_data["from_lat"].max()
    max_lon = bike_data["from_lon"].max()
    min_lon = bike_data["from_lon"].min()
    return min_lat, max_lat, min_lon, max_lon

def latlon_to_pixel(lat, lon, image_shape, bounds):
    min_lat, max_lat, min_lon, max_lon = bounds

    # longitude to pixel conversion (fit data to image)
    delta_x = image_shape[1]/(max_lon-min_lon)

    # latitude to pixel conversion (maintain aspect ratio)
    delta_y = delta_x/np.cos(lat/360*np.pi*2)
    pixel_y = (max_lat-lat)*delta_y
    pixel_x = (lon-min_lon)*delta_x
    return (pixel_y,pixel_x)

def get_path_progress(trips, image_time):
    """ return a series of numbers between 0 and 1
    indicating the progress of each trip at the given time"""
    trip_duration = trips["time_progress"] - trips["start_time"]
    path_progress = (image_time - trips["start_time"]).dt.total_seconds() / trip_duration.dt.total_seconds()
    return path_progress

def get_current_position(trips, progress):
    """ Return Latitude and Longitude for the 'current position' of each trip.
    Paths are assumed to be straight lines between start and end.
    """
    progresso = progress.values[0]
    current_latitude = trips["from_lat"]*(1- progresso) + trips["to_lat"]* progresso
    current_longitude = trips["from_lon"]*(1- progresso) + trips["to_lon"]* progresso
    return current_latitude, current_longitude

def get_active_trips(image_time, bike_data, image_shape, line_len = .1):
    """ Return pixel coordinates only for trips that have started and not yet completed for the given time. """
    bounds = get_min_max(bike_data)
    #active_trips = bike_data[(bike_data["datetime"]<=image_time)]
    active_trips = bike_data[(bike_data["prev_time_progress"]<=image_time)]
    active_trips = active_trips[(active_trips["stop_time"]>=image_time)]
    active_trips_last = active_trips.groupby(["path_id"]).last()
    progress = get_path_progress(active_trips_last, image_time)
    current_latitude, current_longitude = get_current_position(active_trips_last, progress)
    end_y, end_x = latlon_to_pixel(current_latitude, current_longitude, image_shape, bounds)
    group = active_trips.groupby(["path_id"])
    xys = []
    flag = 0
    for name, x in group:
        #x.drop(x.tail(1).index,inplace=True)
        lat = x["from_lat"]
        lon = x["from_lon"]
        start_latitude, start_longitude = get_current_position(x, np.clip(progress-line_len, 0, 1))
        start_y, start_x = latlon_to_pixel(start_latitude, start_longitude, image_shape, bounds)
        data = pd.concat([pd.DataFrame(start_y).rename(columns={0:"from_lat"}),
                          pd.DataFrame(start_x).rename(columns={0:"from_lon"})],
                         axis = 1).reset_index(drop=True)
        a = []
        for i in range(data.shape[0]):
            a.append(data["from_lon"][i])
            a.append(data["from_lat"][i])
        a.append(end_x.iloc[flag])
        a.append(end_y.iloc[flag])
        a = tuple(a)
        xys.append(a)
        flag = flag +1
    weights = np.clip((1 - progress.values)*100, 0, 1)
    return xys, weights

def get_image_map(frame_time, bike_data, city):
    """ Create the folium map for the given time """
    image_data = np.zeros((1000*2,500*2)) #(900*2,400*2)
    bounds = get_min_max(bike_data)
    # plot the current locations
    x, weights = get_active_trips(frame_time, bike_data, image_data.shape, line_len=.01)
    current = []
    for i in range(len(x)):
        cur = x[i][(len(x[i])-4):len(x[i])]
        current.append(cur)
    image_data = add_lines(image_data, current, weights=weights*20, width = 6)
    #  plot the paths
    path_xys, weights = get_active_trips(frame_time, bike_data, image_data.shape, line_len=1)
    image_data = add_lines(image_data, path_xys, weights=weights*10, width = 2)
    # generate and return the folium map.
    return create_image_map(image_data, bounds, city)

def create_image_map(image_data, bounds, city):
    min_lat, max_lat, min_lon, max_lon = bounds
    geolocator = Nominatim(user_agent="visualize")
    location = geolocator.geocode(city)
    folium_map = folium.Map(location=[location.latitude, location.longitude],
                            zoom_start=13,
                            #tiles="cartodbpositron",
                            #tiles="CartoDB dark_matter",
                            width='100%')
    # create the overlay
    map_overlay = add_alpha(to_image(image_data))
    # compute extent of image in lat/lon
    aspect_ratio = map_overlay.shape[1]/map_overlay.shape[0]
    delta_lat = (max_lon-min_lon)/aspect_ratio*np.cos(min_lat/360*2*np.pi)
    # add the image to the map
    img = folium.raster_layers.ImageOverlay(map_overlay,
                               bounds = [(max_lat-delta_lat,min_lon),(max_lat,max_lon)],
                               opacity = 1,
                               name = "Paths")
    img.add_to(folium_map)
    folium.LayerControl().add_to(folium_map)
    # return the map
    return folium_map

def add_lines(image_array, xys, width=1, weights=None):
    for i, xy in enumerate(xys):  # loop over lines
        # create a new gray scale image
        image = Image.new("L",(image_array.shape[1], image_array.shape[0]))
        # draw the line
        ImageDraw.Draw(image).line(xy, 200, width=width)
        #convert to array
        new_image_array = np.asarray(image, dtype=np.uint8).astype(float)
        # apply weights if provided
        if weights is not None:
            new_image_array *= weights[i]
        # add to existing array
        image_array += new_image_array
    # convolve image
    new_image_array = scipy.ndimage.filters.convolve(image_array, get_kernel(width*4))
    return new_image_array

def get_kernel(kernel_size, blur=1/20, halo=.001):
    """
    Create an (n*2+1)x(n*2+1) numpy array.
    Output can be used as the kernel for convolution.
    """
    # generate x and y grids
    x, y = np.mgrid[0:kernel_size*2+1, 0:kernel_size*2+1]
    center = kernel_size + 1  # center pixel
    r = np.sqrt((x - center)**2 + (y - center)**2)  # distance from center
    # now compute the kernel. This function is a bit arbitrary.
    # adjust this to get the effect you want.
    kernel = np.exp(-r/kernel_size/blur) + (1 - r/r[center,0]).clip(0)*halo
    return kernel

def add_alpha(image_data):
    """
    Uses the Value in HSV as an alpha channel.
    This creates an image that blends nicely with a black background.
    """
    # get hsv image
    hsv = rgb_to_hsv(image_data[:,:,:3].astype(float)/255)
    # create new image and set alpha channel
    new_image_data = np.zeros(image_data.shape)
    new_image_data[:,:,3] = hsv[:,:,2]
    # set value of hsv image to either 0 or 1.
    hsv[:,:,2] = np.where(hsv[:,:,2]>0, 1, 0)
    # combine alpha and new rgb
    new_image_data[:,:,:3] = hsv_to_rgb(hsv)
    return new_image_data

# red molto
def to_image(array, hue=.01):
    """converts an array of floats to an array of RGB values using a colormap"""
    # apply saturation function
    image_data = np.log(array + 1)

    # create colormap, change these values to adjust to look of your plot
    saturation_values = [[0, 0], [1, .68], [.78, .87], [0, 1]]
    colors = [hsv_to_rgb([hue, x, y]) for x, y in saturation_values]
    cmap = LinearSegmentedColormap.from_list("my_colormap", colors)

    # apply colormap
    out = cmap(image_data/image_data.max())

    # convert to 8-bit unsigned integer
    out = (out*255).astype(np.uint8)
    return out

def go_paths_frame(params,data, city):
    """Similar to go_arrivals_frame.
    Generate the image, add annotations, and save image file."""
    i, frame_time = params

    my_frame = get_image_map(frame_time, data, city)
    png = my_frame._to_png()

    image = Image.open(io.BytesIO(png))
    draw = ImageDraw.ImageDraw(image)
    font = ImageFont.truetype("Roboto-Light.ttf", 30)

    draw.text((20,image.height - 50),
              "time: {}".format(frame_time),
              fill=(0, 0, 0),
              font=font)

    # draw title
    draw.text((image.width - 450,20),
              "Paths of Individual Trips",
              fill=(0, 0, 0),
              font=font)

    # write to a png file
    dir_name = "path_frames"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    image.save(os.path.join(dir_name, "frame_{:0>5}.png".format(i)), "PNG")
    return image


def visualize(raw,number,second,city):
	df = raw[["path_id","from_lon","from_lat","to_lon","to_lat","distance","start_time",
          "stop_time","start_longitude","stop_longitude","start_latitude","stop_latitude",
          "ID","trip_duration","distance_tot","mean_velocityms","transfer_time",
          "cumulative_transfer_time","time_progress"]]
	df_finale = pd.DataFrame()
	ind = df.groupby(["path_id"])
	for name,x in ind:
		x["prev_time_progress"] = x["time_progress"].shift(1)
		df_finale = pd.concat([df_finale, x], ignore_index=True)
	df_finale["prev_time_progress"] = df_finale.apply(lambda x: substitute(x['prev_time_progress'],x['start_time']), axis = 1)

	start_time = df_finale["start_time"].min()

	frame_times = [(i, start_time + timedelta(seconds=second*i)) for i in range(int(number))]

	t0 = time.time()
	for i in frame_times:
		go_paths_frame(i,df_finale,city)

	print("time elapsed: {}".format(time.time()-t0))
