library("MALDIquantForeign")

spectra = importBrukerFlex('ICC_use',massRange=c(500,1000))

any(sapply(spectra, isEmpty))

table(sapply(spectra, length))

all(sapply(spectra, isRegular))

plot(spectra[[1]])

spectra <- transformIntensity(spectra, method="sqrt")

spectra <- smoothIntensity(spectra, method="SavitzkyGolay",halfWindowSize=10)

baseline <- estimateBaseline(spectra[[1]], method="SNIP",iterations=100)
plot(spectra[[1]])
lines(baseline, col="red", lwd=2)

spectra <- removeBaseline(spectra, method="SNIP",iterations=100)
plot(spectra[[1]])

spectra <- calibrateIntensity(spectra, method="TIC")

spectra <- alignSpectra(spectra,halfWindowSize=20,SNR=3,tolerance=0.002,
                        warpingMethod="lowess")

samples <- factor(sapply(spectra,function(x)metaData(x)$sampleName))

avgSpectra <- averageMassSpectra(spectra, labels=samples,method="mean")

noise <- estimateNoise(avgSpectra[[1]])
plot(avgSpectra[[1]], xlim=c(500, 1000), ylim=c(0, 0.01))
lines(noise, col="red")
lines(noise[,1], noise[, 2]*2, col="blue")

peaks <- detectPeaks(spectra, method="MAD",halfWindowSize=20, SNR=3)

#examine few picked peaks
plot(spectra[[1]], xlim=c(500, 1000), ylim=c(0, 0.01))
points(peaks[[1]], col="red", pch=4)


peaks <- binPeaks(peaks, tolerance=0.002)

peaks <- filterPeaks(peaks, minFrequency=0.02)
featureMatrix <- intensityMatrix(peaks)
featureMatrix_imp <- intensityMatrix(peaks,spectra)

spot <- c()
for(i in 1:length(spectra)){spot[i]<-spectra[[i]]@metaData$spot}
row.names(featureMatrix)<- spot
row.names(featureMatrix_imp)<- spot
write.csv(featureMatrix,file='intensity_tic_pp.csv')
write.csv(featureMatrix_imp,file='intensity_tic_imp_pp.csv')


