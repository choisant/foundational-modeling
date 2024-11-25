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

# Positional arguments: metadata file, traindata file, testdata file
args <- parse_args(parser, positional_arguments = 3)
opt <- args$options

# Read inputs
ntrain <- opt$trainpoints 
nsamples <- opt$samples
nchains <- opt$chains
ncores <- opt$cores


# Read Input arguments
metadatafile <- args$args[1]
traindatafile <- args$args[2]
testdatafile <- args$args[3]

#Test file
test_df <- read.csv(testdatafile)
#test_df <- test_df[1:ntest, ]

# Analysis folder

parent_dir <- dirname(metadatafile)
inferno_dir <- paste0(parent_dir, "/inference/", sub('.csv$', '', basename(traindatafile)), 
                        "/nsamples-", nsamples, "_nchains-",nchains, "_ndata-", ntrain)

##########################################
# RUN TEST
##########################################

# Classes
labels <- cbind(color = c("red", "green"))
nlabels <- length(labels)
xvalues <- test_df[c("x1", "x2")]
ntest <- dim(xvalues)[1]
if (nsamples > 100) {
	nsamples_save <- 100
} else {nsamples_save <- nsamples}
        
Pr_output <- Pr(Y = labels,
                X = xvalues,
                learnt = inferno_dir,
                nsamples = nsamples_save,
                parallel = ncores,
                silent = FALSE)

condfreqs <- Pr_output$samples
quantiles <- Pr_output$quantiles
# Reshape the array from (nlabels, ntest, nsamples) to (nlabels, nsamples, ntest)
condfreqs <- aperm(condfreqs, c(1, 3, 2))
# Reshape the array from (nlabels, ntest, quantiles) to (nlabels, quantiles, ntest)
quantiles <- aperm(quantiles, c(1, 3, 2))

##########################################
# SAVE RESULTS
##########################################
# Create hdf5 file in outputdir
h5file <- file.path(inferno_dir, paste0(sub('.csv$', '', basename(testdatafile)), '_inferred.h5'))
# Overwrite if it already exists
if (!file.exists(h5file)) {
  h5createFile(h5file)
} else {
  file.remove((h5file))
  h5createFile(h5file)
}

# When these files are read in a C-program, the dimensions will be inverted
h5createDataset(h5file, 'probabilities', dims = c(nlabels, ntest))
h5createDataset(h5file, 'quantiles', dims = c(nlabels, 4, ntest))
h5createDataset(h5file, 'samples', dims = c(nlabels, nsamples_save, ntest))
h5createDataset(h5file, 'data', dims = c(nlabels, ntest))
if ("color" %in% names(test_df)){
        h5createDataset(h5file, 'truth', dims = c(ntest))
}
# Write to file
cat('Writing to file \n')
h5write(Pr_output$values, file = h5file, name = 'probabilities',
        index = list(NULL, NULL))
h5write(quantiles, file = h5file, name = 'quantiles',
        index = list(NULL, NULL, NULL))
h5write(condfreqs, file = h5file, name = 'samples',
        index = list(NULL, NULL, NULL))
h5write(t(test_df[c("x1", "x2")]),
        file = h5file, name = 'data', index = list(NULL, NULL))
if ("color" %in% names(test_df)) {
        h5write(t(test_df["color"]), file = h5file, name = 'truth',
                index = list(NULL))
}

#For testing, let's plot the first test point to see that we can reproduce it in python

#pdf(paste0(inferno_dir, "/test_plot.pdf"))
#plot(sapply(Pr_output,'[[',1))