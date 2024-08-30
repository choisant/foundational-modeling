library("inferno")
set.seed(120)
inferno_dir <- "inferno"
if(!dir.exists(inferno_dir)) {dir.create((inferno_dir))}
caldata <- "data/train.csv"
#buildmetadata(caldata, file = "inferno/temp_metadata")
metadata <- "inferno/metadata.csv"

NSAMPLES <- 32
NCHAINS <- 4
NTRAIN <- 50
NCORES <- 4

#Sample train data
alldata <- data.table::fread(caldata)
trainpoints <- sort(sample(seq_len(nrow(alldata)), NTRAIN))

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

probability <- Pr(
    Y = data.frame(diff.MDS.UPRS.III = 0),
    X = data.frame(Sex = 'Female', TreatmentGroup = 'NR'),
    learnt = learnt
)