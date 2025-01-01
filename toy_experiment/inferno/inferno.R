library("inferno")
library('rhdf5')
library('optparse')

##########################################
# SETUP
##########################################

# Usage
# Rscript inferno_test.R <options> metadatafile trainfile testfile

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

parser <- add_option(parser, c("--test"),  default = "None",
        help = "Test data CSV file. If not provided, no tests will be run.")

parser <- add_option(parser, c("--runLearn"),
        type = "logical", default = FALSE,
        help = "Boolean, whether or not to run MC sampling [default %default]")

# Positional arguments: metadata file, traindata file
args <- parse_args(parser, positional_arguments = 2)
opt <- args$options

# Read inputs
metadatafile <- args$args[1]
traindatafile <- args$args[2]

ntrain <- opt$trainpoints 
nsamples <- opt$samples
nchains <- opt$chains
ncores <- opt$cores
testdatafile <- opt$test
runLearn <- opt$runLearn

# Read files
if (file.exists(traindatafile)) {
        train_df <- read.csv(traindatafile)
    } else {stop("Please provide a valid CSV file as training data.")}

# Data is already shuffled, no need to reshuffle.
traindata <- train_df[1:ntrain, ]

if (testdatafile != "None") {
	if (file.exists(testdatafile)) {
        test_df <- read.csv(testdatafile)
    } else {stop("Please provide a valid CSV file as test data.")}
}

# Check if polar coordinates
if (is.character(metadatafile) && file.exists(metadatafile)) {
        metadata <- read.csv(metadatafile, na.strings = '')
    } else {stop("Please provide a valid CSV file as metadata.")}

if (all(c("r_x", "a_x") %in% metadata[['name']])) {
	polar <- TRUE
} else polar <- FALSE

#ncheckpoints <- Inf # Use all datapoints to check for convergence
if (ntrain > 16) {
	ncheckpoints <- 20 
} else ncheckpoints <- Inf
        
# Create inference folder in same folder as metadata file
parent_dir <- dirname(metadatafile)
# Subfolder: traindatafile/nsamples-X_nchains-Y_ndata-Z
inferno_dir <- paste0(parent_dir, "/inference/", sub('.csv$', '', basename(traindatafile)), 
                        "/nsamples-", nsamples, "_nchains-",nchains, "_ndata-", ntrain)
if(!dir.exists(inferno_dir) && runLearn == TRUE) {
	cat(paste0("Creating dir ", inferno_dir, '\n'))
	dir.create(inferno_dir, recursive=TRUE)
}

##########################################
# TRAIN MODEL
##########################################

# Start inference
if (runLearn) {
	cat("Starting MC simulation. \n")
	learnt <- learn(
		data = traindata,
		metadata = metadatafile,
		outputdir = inferno_dir,
		nsamples = nsamples,
		nchains = nchains,
		ncheckpoints = ncheckpoints,
		parallel = ncores,
		appendtimestamp = FALSE,
		appendinfo = FALSE
	)
} else {cat("Not running MC simulation. \n")}

##########################################
# RUN TEST
##########################################
if (testdatafile != "None") {
	# Classes
	labels <- cbind(class = c(0, 1))
	nlabels <- length(labels)
	if (polar) {
		xvalues <- test_df[c("r_x", "a_x")]
	} else {
		xvalues <- test_df[c("x1", "x2")]
	}
	nxvalues <- 2

	ntest <- dim(xvalues)[1]
	if (nsamples > 100) {
		nsamples_save <- 100
	} else {nsamples_save <- nsamples}

	starttime <- Sys.time()
	cat("Starting inference.")
			
	Pr_output <- Pr(Y = labels,
					X = xvalues,
					learnt = inferno_dir,
					nsamples = nsamples_save,
					parallel = ncores,
					silent = FALSE)


	cat("End inference. \n")

	printdifftime <- function(time1, time2) {
			difference = difftime(time1, time2, units = 'auto')
			paste0(signif(difference, 2), ' ', attr(difference, 'units'))
		}

	cat(paste0("Total time for inference: ", printdifftime(Sys.time(), starttime), '\n'))


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
	h5createDataset(h5file, 'data', dims = c(nxvalues, ntest))
	if ("class" %in% names(test_df)){
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
	if (polar) {
		h5write(t(test_df[c("r_x", "a_x")]),
			file = h5file, name = 'data', index = list(NULL, NULL)
			)
	} else {
		h5write(t(test_df[c("x1", "x2")]),
			file = h5file, name = 'data', index = list(NULL, NULL))
	}
	if ("class" %in% names(test_df)) {
			h5write(t(test_df["class"]), file = h5file, name = 'truth',
					index = list(NULL))
	}
} else {cat("Not running any tests. \n")}