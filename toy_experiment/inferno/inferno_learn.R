library("inferno")
library('rhdf5')
library('optparse')

##########################################
# SETUP
##########################################

# This script can be used with an argument specifying the number of data points to train the model oncores
# It should be used with the path to the training data and the test data
# Read and parse arguments

parser <- OptionParser(usage="%prog [options] trainfile testfile")

parser <- add_option(parser, c("-N", "--trainpoints"),
        type = "integer", default = 5,
        help = "Number of data points to train on [default %default]")

parser <- add_option(parser, c("-c", "--chains"),
        type = "integer", default = 1,
        help = "Number of MCMC chains [default %default]")

parser <- add_option(parser, c("-p", "--cores"),
        type = "integer", default = 1,
        help = "Number of cores to use for parallel processing [default %default]")

parser <- add_option(parser, c("-s", "--samples"),
        type = "integer", default = 120,
        help = "Number of MCMC samples to generate [default %default]")  

# Positional arguments: metadata file, traindata file
args <- parse_args(parser, positional_arguments = 3)
opt <- args$options

# Read inputs
ntrain <- opt$trainpoints 
nsamples <- opt$samples
nchains <- opt$chains
ncores <- opt$cores

#ncheckpoints <- Inf # Use all datapoints to check for convergence
#Apparently doesn't work
if (ntrain > 16) {
	ncheckpoints <- ntrain %/% 4 # Use 25 % of training data for checkpoints
} else {ncheckpoints <- Inf}


# Read files
metadatafile <- args$args[1]
traindatafile <- args$args[2]
testdatafile <- args$args[3]

train_df <- read.csv(traindatafile)
# Data is already shuffled, no need to reshuffle
traindata <- train_df[1:ntrain, ]
test_df <- read.csv(testdatafile)

# Create folder
inferno_dir <- paste0("analysis/", sub('.csv$', '', basename(traindatafile)), 
                        "-nsamples_", nsamples, "_ndata_", ntrain)
if(!dir.exists(inferno_dir)) {
	cat(paste0("Creating dir ", inferno_dir, '\n'))
	dir.create(inferno_dir, recursive=TRUE)
    }

##########################################
# TRAIN MODEL
##########################################

# Start inference
learnt <- learn(
    data = traindata,
    metadata = metadatafile,
    outputdir = inferno_dir,
    nsamples = nsamples,
    nchains = nchains,
    ncheckpoints = 20,
    parallel = ncores,
    appendtimestamp = FALSE,
    appendinfo = FALSE
)
