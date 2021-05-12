library(mvtnorm)
get_proba <- function(filename_Tau, filename_Omega, maxpts_=1000) {
	Tau <- read.delim(filename_Tau, header=FALSE)
	Omega <- read.delim(filename_Omega, header=FALSE, sep=" ")
	Omega_shape = dim(Omega)
	Omega <- matrix(unlist(Omega), ncol=Omega_shape[2], nrow=Omega_shape[1])
	Cov = Omega %*% t(Omega)
	Tau = as.numeric(unlist(Tau))
	ans = 1 - pmvnorm(mean=rep(0,Omega_shape[1]), sigma=Cov, lower=rep(-Inf,Omega_shape[1]), upper=Tau, maxpts=maxpts_)
 	return(ans)
}
compute_probas <- function(prefix) {
	pattern_Omega = paste("^.*", prefix, ".*.Omega", sep="")
	pattern_Tau = paste("^.*", prefix, ".*.Tau", sep="")
	files_Omega = sort(list.files(pattern=pattern_Omega))
	files_Tau = sort(list.files(pattern=pattern_Tau))
	n_files = length(files_Omega)
	out = list()
	for(i in 1:n_files){
		proba = get_proba(files_Tau[i], files_Omega[i])[1]
		out[[i]] <- proba
	}
	write.table(out, file=paste(prefix, "_R_results.txt", sep=""), sep="\t") 
	#capture.output(summary(out), file = paste(prefix, "results.txt", sep=""))
}
get_data_for_table <- function(){
  #data_prefix_list = list('case118ipi2', 'case118ipi3', 'case118ipi4', 'case118ipi6')
  #data_prefix_list = c(data_prefix_list, list('case57pi2', 'case57pi3', 'case57pi4', 'case57pi6'))
  data_prefix_list = list('case200pi2', 'case200pi3', 'case200pi4', 'case200pi6', 'case200pi7', 'case200pi8')
  data_prefix_list = c(data_prefix_list, list('case30pi2', 'case30pi3', 'case30pi4', 'case30pi6', 'case30pi7', 'case30pi8'))
  for(prefix in data_prefix_list){
    print(prefix)
    get_exp_plot_data(prefix)  
  }
  
}
get_exp_plot_data <- function(prefix) {
  pattern_Omega = paste("^.*", prefix, ".*.Omega", sep="")
  pattern_Tau = paste("^.*", prefix, ".*.Tau", sep="")
  files_Omega = sort(list.files(pattern=pattern_Omega))
  files_Tau = sort(list.files(pattern=pattern_Tau))
  n_files = length(files_Omega)
  out = list()
  for(i in 1:1000){
    proba = get_proba(files_Tau[1], files_Omega[1], maxpts_ = i)[1]
    out[i] = proba
  }
  
  write.table(out, file=paste(prefix, "exp_results.txt", sep=""), sep="\t") 
  
}
prepare_hist <- function(filename_Tau, filename_Omega, maxpts_=1000, n_tries=100){
	out = list()
	for(i in 1:n_tries){
		ans = get_proba(filename_Tau, filename_Omega, maxpts=maxpts_)
		out[[i]] = ans / exp(-6 * 6 / 2)#0.31731050786291404#
	}
	write.table(out, file="hist_data.txt", row.names=FALSE, sep="\t", col.names=FALSE)
}