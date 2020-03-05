# packages
library(osmdata)
library(polyclip)
library(sf)
library(dodgr)
library(vroom)
library(dplyr)
library(tidyr)
library(janitor)
library(lubridate)
library(geodist)
library(tibble)

assign("last.warning",NULL,envir = baseenv())
args <- commandArgs(trailingOnly = TRUE)
if (length(args)!=2) {
  stop("The required arguments are two, the csv cointaining start and end point and the type of vehicle", call.=FALSE)
} 
filename <- args[1]
vehicle_type <- args[2]

data_ridotto <- vroom(filename) %>% clean_names()

start <- data.frame(x = data_ridotto$start_longitude,
                    y = data_ridotto$start_latitude)
end <- data.frame(x = data_ridotto$stop_longitude,
                  y = data_ridotto$stop_latitude)

# the bounding box is enlarged about one kilometer for each direction
max_lat <- max(start$y,end$y) + 0.03
min_lat <- min(start$y,end$y) - 0.03
max_lon <- max(start$x,end$x) + 0.05
min_lon <- min(start$x,end$x) - 0.05

limits <- rbind(c(max_lon, max_lat),c(max_lon, min_lat), c(min_lon, min_lat), c(min_lon, max_lat))

city_streetnet <- dodgr_streetnet(bbox = limits, quiet = FALSE)


city_weighted <- weight_streetnet(city_streetnet, wt_profile = vehicle_type)

# apply filter
city_weighted_filtered <- city_weighted#[check, ]


lista_path <- list()
for(i in 1:nrow(data_ridotto)) {
  inizio <- c(start[i,1],start[i,2])
  fine <- c(end[i,1],end[i,2])
  path <- dodgr_paths(city_weighted_filtered, from = inizio, to = fine, vertices = FALSE)[[1]][[1]]
  lista_path[[i]] <- path
}

df <- tibble()
flag <-  1
for(i in 1:length(lista_path)){
  for(j in 1:length(lista_path[[i]])){
    if(!is.null(lista_path[[i]][j])){
      df[flag,"path_id"] <- i
      df[flag,"from_lon"]  <- city_weighted_filtered[lista_path[[i]][j],"from_lon" ]
      df[flag,"from_lat"]  <- city_weighted_filtered[lista_path[[i]][j],"from_lat" ]
      df[flag,"to_lon"]  <- city_weighted_filtered[lista_path[[i]][j],"to_lon" ]
      df[flag,"to_lat"]  <- city_weighted_filtered[lista_path[[i]][j],"to_lat" ]
      df[flag,"calculated"]  <- TRUE
      flag <- flag + 1
    } else if (j == 0) {
    } else {
      df[flag,"path_id"] <- i
      df[flag,"from_lon"]  <- 9.99
      df[flag,"from_lat"]  <- 9.99
      df[flag,"to_lon"]  <- 9.99
      df[flag,"to_lat"]  <- 9.99
      df[flag,"calculated"]  <- FALSE
      flag <- flag + 1
    }
  }
}

write.table(df , file = "path.csv", sep = ",", col.names = TRUE, row.names = FALSE)






