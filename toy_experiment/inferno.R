library("inferno")
set.seed(120)
inferno_dir <- "inferno/run"
if(!dir.exists(inferno_dir)) {dir.create((inferno_dir))}
caldata <- "data/cal.csv"
testdata <- "data/test.csv"
#buildmetadata(caldata, file = "inferno/temp_metadata")
metadata <- "inferno/metadata.csv"

NSAMPLES <- 32
NCHAINS <- 1
NTRAIN <- 10
NCORES <- 1


# Start inference
learnt <- learn(
    data = caldata,
    metadata = metadata,
    outputdir = inferno_dir,
    nsamples = NSAMPLES,
    nchains = NCHAINS,
    parallel = NCORES,
    appendtimestamp = FALSE,
    appendinfo = FALSE
)
