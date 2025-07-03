library("inferno")
library('rhdf5')
library('optparse')

##########################################
# SETUP
##########################################

hyperparams = list(
        ncomponents = 300,
        minalpha = -4,
        maxalpha = 4,
        byalpha = 1,
        Rshapelo = 0.5,
        Rshapehi = 0.5,
        Rvarm1 = 3^2,
        Cshapelo = 0.5,
        Cshapehi = 0.5,
        Cvarm1 = 3^2,
        Dshapelo = 0.5,
        Dshapehi = 0.5,
        Dvarm1 = 3^2,
        Lshapelo = 0.5,
        Lshapehi = 0.5,
        Lvarm1 = 3^2,
		# Bayes-Laplace prior for Bernoulli distribution
        Bshapelo = 1,
        Bshapehi = 1,
        Dthreshold = 1
    )

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

# Don't shuffle data
traindata <- train_df[1:ntrain, ]

if (testdatafile != "None") {
	if (file.exists(testdatafile)) {
        test_df <- read.csv(testdatafile)
    } else {stop("Please provide a valid CSV file as test data.")}
}

if (is.character(metadatafile) && file.exists(metadatafile)) {
        metadata <- read.csv(metadatafile, na.strings = '')
    } else {stop("Please provide a valid CSV file as metadata.")}

# Check metadata file for polar coordinates
if (all(c("r", "a1") %in% metadata[['name']])) {
	polar <- TRUE
} else polar <- FALSE

#ncheckpoints <- Inf # Use all datapoints to check for convergence
if (ntrain > 16) {
	ncheckpoints <- 20 
} else ncheckpoints <- Inf
        
# Create inference folder in same folder as metadata file
parent_dir <- dirname(metadatafile)
# Subfolder: traindatafile/nsamples-X_nchains-Y_ndata-Z
if (polar) {
	inferno_dir <- paste0(parent_dir, "/inference/", sub('.csv$', '', basename(traindatafile)), 
                        "/nsamples-", nsamples, "_nchains-",nchains, "_ndata-", ntrain, "_POLAR")					
} else if ((hyperparams$Bshapehi != 1) && (hyperparams$Bshapelo != 1)) {
   inferno_dir <- paste0(parent_dir, "/inference/", sub('.csv$', '', basename(traindatafile)), 
                        "/nsamples-", nsamples, "_nchains-",nchains, "_ndata-", ntrain, "_Beta_", 
						hyperparams$Bshapehi, "_", hyperparams$Bshapelo)
} else {
	inferno_dir <- paste0(parent_dir, "/inference/", sub('.csv$', '', basename(traindatafile)), 
                        "/nsamples-", nsamples, "_nchains-",nchains, "_ndata", ntrain, "_ncomps_",
						hyperparams$ncomponents)						
}

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
		appendinfo = FALSE,
		hyperparams = hyperparams,
	)
	summary(warnings())
} else {cat("Not running MC simulation. \n")}

##########################################
# RUN TEST
##########################################
if (testdatafile != "None") {
	# Run inference
	starttime <- Sys.time()
	cat("Starting inference.")
	# Class can be 0 or 1
	labels <- cbind(class = c(0, 1))
	# Number of classes
	nlabels <- length(labels)
	# Crashes for too many data points. Do n_samples_per_file at a time.
	length_df <- dim(test_df)[1]
	processed <- 0
	n_samples_per_file <- 1000
	# Keep going until all datapoints are processed
	while (processed < length_df) {
		# Check if we should add n_samples_per_file or just the rest of the data
		if ((length_df - processed) > n_samples_per_file) {
			i_start <- processed + 1
			i_end <- processed + n_samples_per_file
			testdata <- test_df[(i_start):(i_end), ]
			if (polar) {
				xvalues <- testdata[c("r_x", "a_x")]
			} else {
				xvalues <- testdata[c("x1", "x2")]
			}
		} else {
			i_start <- processed + 1
			i_end <- length_df
			testdata <- test_df[(i_start):(i_end), ]
			if (polar) {
				xvalues <- testdata[c("r_x", "a_x")]
			} else {
				xvalues <- testdata[c("x1", "x2")]
			}
		}
		# Number of non class-dimensions
		nxvariates <- 2
		# Number of data points to test
		ntest <- dim(xvalues)[1]
		# Takes forever to process all samples, reduce nr of samples for uncertainty calculations
		nsamples_max <- 1200
		if (nsamples > nsamples_max) {
			nsamples_save <- nsamples_max
		} else {nsamples_save <- nsamples}

				
		Pr_output <- Pr(Y = labels,
						X = xvalues,
						learnt = inferno_dir,
						nsamples = nsamples_save,
						parallel = ncores,
						silent = FALSE)

		cat("End inference. \n")
		
		# Get values we need
		stds <- apply(Pr_output$samples, c(1,2), sd) #output (nlabels, ntest)
		quantiles <- Pr_output$quantiles
		# Reshape the array from (nlabels, ntest, quantiles) to (nlabels, quantiles, ntest)
		quantiles <- aperm(quantiles, c(1, 3, 2))

		##########################################
		# SAVE RESULTS
		##########################################
		# Create hdf5 file in outputdir
		h5file <- file.path(inferno_dir, paste0(sub('.csv$', '', basename(testdatafile)),'_', i_start,
							'-', i_end,'_inferred.h5'))
		# Overwrite if it already exists
		if (!file.exists(h5file)) {
		h5createFile(h5file)
		} else {
		file.remove((h5file))
		h5createFile(h5file)
		}
		# When these files are read in a C-program, the dimensions will be inverted
		h5createDataset(h5file, 'probabilities', dims = c(nlabels, ntest))
		h5createDataset(h5file, 'stds', dims = c(nlabels, ntest))
		h5createDataset(h5file, 'quantiles', dims = c(nlabels, 4, ntest))
		h5createDataset(h5file, 'data', dims = c(nxvariates, ntest))
		if ("class" %in% names(testdata)){
				h5createDataset(h5file, 'truth', dims = c(ntest))
		}
		# Write to file
		cat('Writing to file \n')
		h5write(Pr_output$values, file = h5file, name = 'probabilities',
				index = list(NULL, NULL))
		h5write(stds, file = h5file, name = 'stds',
				index = list(NULL, NULL))
		h5write(quantiles, file = h5file, name = 'quantiles',
				index = list(NULL, NULL, NULL))
		if (polar) {
			h5write(t(testdata[c("r", "a1")]),
				file = h5file, name = 'data', index = list(NULL, NULL)
				)
		} else {
			h5write(t(testdata[c("x1", "x2")]),
				file = h5file, name = 'data', index = list(NULL, NULL))
		}
		if ("class" %in% names(testdata)) {
				h5write(t(testdata["class"]), file = h5file, name = 'truth',
						index = list(NULL))
		}
		# update processed variable
		processed <- processed + ntest
		# Warnings?
		summary(warnings())
	}
	# How long did it take?
	printdifftime <- function(time1, time2) {
			difference = difftime(time1, time2, units = 'auto')
			paste0(signif(difference, 2), ' ', attr(difference, 'units'))
		}

	cat(paste0("Total time for inference: ", printdifftime(Sys.time(), starttime), '\n'))
} else {cat("Not running any tests. \n")}