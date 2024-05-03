
X <- matrix(c(1,2,2,4,3,9), nrow = 3, ncol = 2, byrow=T)
y = c(1,2,3)

a <- ncvreg::ncvreg(X, y)
