import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #DONE 12/22/2016  
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
      scores = X[i,:].dot(W)
      scores_min = np.min(scores)
      #print "min = {}".format(scores_min)
      if scores_min <0:
          C = -scores_min
      else:
          C = 0
      nom = np.exp(scores[y[i]] + C)
      denom = np.sum(  np.exp(scores + C) )
      for j in xrange(num_class):
          if j != y[i]:
              dW[:, j] += np.exp(scores[j]+C) / denom * X[i]
          else:
              dW[:,j] += np.exp(scores[j]+C) / denom * X[i] - X[i]
      loss += -np.log(nom/denom)
  loss += 0.5 * reg * np.sum(W * W)
  loss /= num_train
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]  
  num_class = W.shape[1]
  scores = X.dot(W)
  min_score = np.min(scores)
  C = 0
  if min_score < 0:
      C = -min_score
  exp_scores = np.exp( scores + C)
  denom = np.tile(np.sum(exp_scores, axis = 1) , (num_class,1)).T
  prob = exp_scores*1.0 / denom
  dW = np.dot(X.T, prob)
  y_ind = np.zeros((num_train, num_class))
  for i in xrange(len(y)):
      y_ind[i, y[i]] = 1
  loss = np.sum( -np.log(prob) * y_ind  )
  dW -= np.dot( X.T, y_ind)
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

