library(flowCore)
library(FlowSOM)

flowset <- read.flowSet(path = "/Users/gjbaker/Dropbox (HMS)/Baker_et_al_2018/Baker_2019_04_05/cleaned_zeroed_fcs_files/")

fSOM <- FlowSOM(flowset, compensate = FALSE, transform = FALSE, scale = FALSE, colsToUse = c(3:13), nClus = 30, seed = 1)

setwd("/Users/gjbaker/projects/gbm_immunosuppression/flowsom/data")

pdf(file="star_plot.pdf")
PlotStars(fSOM$FlowSOM, backgroundValues = as.factor(fSOM$metaclustering))
dev.off()

# combine metacluster numbers with signal intensity expression data
df <- data.frame(fSOM$metaclustering[fSOM$FlowSOM$map$mapping[,1]], fSOM$FlowSOM$data)

# update column names
colnames(df)[1] <- "cluster"

# save df
write.csv(df,"flowsom.csv")

# delete dataframe to free up memory
rm(df)

# get a list of metadata
path= c()
for (i in 1: length(fSOM$FlowSOM$metaData)){
x = strsplit(names(fSOM$FlowSOM$metaData), '/')[[i]][9]
path = c(path,x)
}

# get a list of row index ranges
range = list()
for (i in 1: length(fSOM$FlowSOM$metaData)){
x = fSOM$FlowSOM$metaData[i][[1]]
range[[i]] <- x
}

# get a one column dataframe of metadata the length of df
metadata = data.frame()
for (i in 1: length(range)){
  val = (range[[i]][2] - range[[i]][1]) + 1
  z = data.frame(rep(path[i],val))
  metadata = rbind(metadata,z)
}

# update metadata column name
names(metadata) <- c("metadata")

# save metadata dataframe
write.csv(metadata,"metadata.csv")
