XfromT <- function(tt, r = 7, R = 3){
    tt <- cbind(tt)
    t1 <- tt[1, ]
    t2 <- tt[2, ]
    ##
    rbind(
        x1 = r * cos(t1) + R * cos(t1 + t2),
        x2 = r * sin(t1) + R * sin(t1 + t2)
    )
}

TfromX <- function(xx, r = 7, R = 3){
    xx <- cbind(xx)
    x1 <- xx[1, ]
    x2 <- xx[2, ]
    x <- sqrt(x1^2 + x2^2)
    t1 <-  2 * atan((
            2 * r * x2 - sqrt(2 * r^2 * (R^2 + x^2) - (x^2 - R^2)^2 - r^4)
        ) / (
            x^2 + r^2 + 2 * r * x1 - R^2
        ))
    ## define different branch for arctan
    temp <- !is.na(t1) & t1 < 0
    t1[temp] <- t1[temp] + 2 * pi
    ##
    rbind(
        t1 = t1,
        t2 = -2 * atan((
            sqrt(2 * r^2 * (R^2 + x^2) - (x^2 - R^2)^2 - r^4)
        ) / (
            (r - R)^2 - x^2
        ))
    )
}

## Check inverse
n <- 100
#samplesR <- 10^(runif(n = n, -2, 2)) # from 0.01 to 100
samplest1 <- runif(n = n, 0, 2 * pi)
samplest2 <- runif(n = n, 0, pi)
#samplesr <- 10^(runif(n = n, log10(2 * samplesR), log10(max(2 * samplesR) + 100)))
## combine
samplestt <-  rbind(t1 = samplest1, t2 = samplest2)
##
## calculate x from theta
samplesxx <- XfromT(samplestt)
## calculate theta from x
samplesinv <- TfromX(samplesxx)

#pdf('test_inverse')
#myflexiplot(x=samplestt[1, ], y=samplesinv[1, ], type='p', pch='.',
#    xlab='theta_1 input', ylab='theta_1 from inverse')
#myflexiplot(x=samplestt[2, ], y=samplesinv[2, ], type='p', pch='.',
#    xlab='theta_2 input', ylab='theta_2 from inverse')
#dev.off()
## ## see some values (uncomment only if n is small)
## samplesttr
## samplesinv
## Check max discrepancy
max(abs(samplestt - samplesinv))
# > [1] 9.80312e-06
