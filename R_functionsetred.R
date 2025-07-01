
function (y, D, gen.learner, gen.pred, theta = 0.1, max.iter = 50, 
          perc.full = 0.7) 
{
  if (!is.factor(y)) {
    if (!is.vector(y)) {
      stop("Parameter y is neither a vector nor a factor.")
    }
    else {
      y = as.factor(y)
    }
  }
  if (inherits(D, "dist")) {
    D <- proxy::as.matrix(D)
  }
  if (!is.matrix(D)) {
    stop("Parameter D is neither a matrix or a dist object.")
  }
  else if (nrow(D) != ncol(D)) {
    stop("The distance matrix D is not a square matrix.")
  }
  else if (nrow(D) != length(y)) {
    stop(sprintf(paste("The dimensions of the matrix D is %i x %i", 
                       "and it's expected %i x %i according to the size of y."), 
                 nrow(D), ncol(D), length(y), length(y)))
  }
  if (!(theta >= 0 && theta <= 1)) {
    stop("theta must be between 0 and 1")
  }
  if (max.iter < 1) {
    stop("Parameter max.iter is less than 1. Expected a value greater than and equal to 1.")
  }
  if (perc.full < 0 || perc.full > 1) {
    stop("Parameter perc.full is not in the range 0 to 1.")
  }
  classes <- levels(y)
  nclasses <- length(classes)
  ynew <- y
  labeled <- which(!is.na(y))
  unlabeled <- which(is.na(y))
  if (length(labeled) == 0) {
    stop("The labeled set is empty. All the values in y parameter are NA.")
  }
  if (length(unlabeled) == 0) {
    stop("The unlabeled set is empty. None value in y parameter is NA.")
  }
  cls.summary <- summary(y[labeled])
  proportion <- cls.summary/length(labeled)
  cantClass <- round(cls.summary/min(cls.summary))
  totalPerIter <- sum(cantClass)
  count.full <- length(labeled) + round(length(unlabeled) * 
                                          perc.full)
  iter <- 1
  while ((length(labeled) < count.full) && (length(unlabeled) >= 
                                            totalPerIter) && (iter <= max.iter)) {
    model <- gen.learner(labeled, ynew[labeled])
    prob <- checkProb(prob = gen.pred(model, unlabeled), 
                      ninstances = length(unlabeled), classes)
    selection <- selectInstances(cantClass, prob)
    nlabeled.old <- length(labeled)
    labeled.prime <- unlabeled[selection$unlabeled.idx]
    sel.classes <- classes[selection$class.idx]
    ynew[labeled.prime] <- sel.classes
    labeled <- c(labeled, labeled.prime)
    unlabeled <- unlabeled[-selection$unlabeled.idx]
    nlabeled.new <- length(labeled)
    ady <- vector("list", nlabeled.new)
    for (i in (nlabeled.old + 1):nlabeled.new) {
      for (j in 1:(i - 1)) {
        con <- TRUE
        for (k in 1:nlabeled.new) if (k != i && k != 
                                      j && D[labeled[i], labeled[j]] > max(D[labeled[i], 
                                                                             labeled[k]], D[labeled[k], labeled[j]])) {
          con <- FALSE
          break
        }
        if (con) {
          ady[[i]] <- c(ady[[i]], j)
          ady[[j]] <- c(ady[[j]], i)
        }
      }
    }
    noise.insts <- c()
    for (i in (nlabeled.old + 1):nlabeled.new) {
      propi <- proportion[unclass(ynew[labeled[i]])]
      Oi <- 0
      nv <- W <- k <- 0
      for (j in ady[[i]]) {
        k <- k + 1
        W[k] <- 1/(1 + D[labeled[i], labeled[j]])
        if (ynew[labeled[i]] != ynew[labeled[j]]) {
          Oi <- Oi + W[k]
          nv <- nv + 1
        }
      }
      if (normalCriterion(theta, Oi, length(ady[[i]]), 
                          propi, W)) {
        noise.insts <- c(noise.insts, i)
      }
    }
    if (length(noise.insts) > 0) {
      ynew[labeled[noise.insts]] <- NA
      labeled <- labeled[-noise.insts]
    }
    iter <- iter + 1
  }
  model <- gen.learner(labeled, ynew[labeled])
  result <- list(model = model, instances.index = labeled)
  class(result) <- "setredG"
  result
}