library("inferno")
set.seed(120)
inferno_dir <- "inferno/run"
if(!dir.exists(inferno_dir)) {dir.create((inferno_dir))}
caldata <- "data/cal.csv"
buildmetadata(caldata, file = "inferno/temp_metadata")
stop()
#metadata <- "inferno/metadata.csv"

NSAMPLES <- 3200
NCHAINS <- 12
NCORES <- 12


# Start inference
learnt <- learn(
    data = caldata,
    metadata = metadata,
    outputdir = inferno_dir,
    nsamples = NSAMPLES,
    nchains = NCHAINS,
    parallel = NCORES,
    appendtimestamp = FALSE,
    appendinfo = TRUE
)
