library('raster')
library('maptools')
library('sf')
library('rgdal')
library('ggplot2')
library('fasterize')
library('tmap')

setwd("~/LDAproject/Code/comb")

# load base map
data("World")

# load different topics
topic_spdf <- st_read("../../Data/mammals_range/mammals_topic_comp.csv",
                      stringsAsFactors = FALSE)
                      # query = "SELECT * FROM mam_topic_comp WHERE Conservati = '1.0' LIMIT 200"

for (i in colnames(topic_spdf)[13:29]){
  topic_spdf[[i]] <- as.numeric(topic_spdf[[i]])
}

# drop columns
topic_spdf <- subset(topic_spdf, select = -c(id_no, Red_list_category, Realm, Systems, Class))

# create a raster_template
rst_temp <- raster(ncols = (640), nrows = (960),
                   crs = "+proj=longlat +datum=WGS84 +no_defs",
                   extent(topic_spdf))

j = 1
for (i in colnames(topic_spdf)[8:24]){
  print(i)
  # fasterize
  rst_count <- fasterize(topic_spdf, rst_temp, field = i, fun = 'count', background = 0)
  rst_sum <- fasterize(topic_spdf, rst_temp, field = i, fun = 'sum', background = 0)
  rst_mean <- rst_sum/rst_count
  # change NA to 0
  rst_mean[is.na(rst_mean)] <- 0
  
  # save raster
  writeRaster(rst_mean, paste0("../../Data/mammals_range/rasters/", j, "_", i), "GTiff", overwrite = TRUE)
  svg(paste0("../../Data/mammals_range/rasters/", j, "_", i, ".svg"), width = 10, height = 5.1)
  g <- tm_shape(rst_mean) +
    tm_raster(style = "fixed",
              breaks = c(0, seq(0.5,1,0.05)),
              palette=c('ivory2', 'lightblue', 'khaki2', 'red3'),
              title = "Proportion of topic\nto species present") +
    tm_legend(outside = FALSE,
              position = c("left", "bottom"),
              outer.margins = c(0.001,0.001,0.01,0.001),
              bg.color='white',
              bg.alpha=0.4,
              scale = 1.25) +
    tm_shape(World) +
    tm_borders("black", lwd = 0.5) +
    tm_layout(outer.margins = c(0.001,0.001,0.01,0.001))
  # got to use print when saving plot inside a loop
  print(g)
  dev.off()
  j = j + 1
}

# plot legend and save


# trying out removing pixels where species count is < 3
i = "Monitor_is"

rst_count <- fasterize(topic_spdf, rst_temp, field = i, fun = 'count', background = 0)

# check count at each pixel, change to zero if less than 3 species
rst_count[rst_count < 3] <- 0

rst_sum <- fasterize(topic_spdf, rst_temp, field = i, fun = 'sum', background = 0)
rst_mean <- rst_sum/rst_count

# division by 0 produces Inf, change back to zero
rst_mean[rst_mean == Inf] <- 0
# change NA to 0
rst_mean[is.na(rst_mean)] <- 0
unique(rst_count@data@values)


g <- tm_shape(rst_mean) +
  tm_raster(style = "fixed",
            breaks = c(0, seq(0.5,1,0.05)),
            palette=c('ivory2', 'lightblue', 'khaki2', 'red3'),
            title = "Proportion of topic\nto species present") +
  tm_legend(outside = FALSE,
            position = c("left", "bottom"),
            outer.margins = c(0.001,0.001,0.01,0.001),
            bg.color='white',
            bg.alpha=0.4,
            scale = 1.25) +
  tm_shape(World) +
  tm_borders("black", lwd = 0.5) +
  tm_layout(outer.margins = c(0.001,0.001,0.01,0.001))
# par(mar=rep(0,4),
    # oma=rep(0,4))
# got to use print when saving plot inside a loop
print(g)



# use tmap_arrange to plot multipanel
tmap_arrange(g,g,g,g)
