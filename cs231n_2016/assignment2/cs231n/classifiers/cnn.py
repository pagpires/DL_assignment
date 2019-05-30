import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

# XH helper functions for batch normalization
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  out, fc_cache = affine_forward(x, w, b)
  out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
  out, relu_cache = relu_forward(out)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  drelu = relu_backward(dout, relu_cache)
  dbn, dgamma, dbeta = batchnorm_backward(drelu, bn_cache)
  dx, dw, db = affine_backward(dbn, fc_cache)
  return dx, dw, db, dgamma, dbeta

def conv_bn_relu_pool_forward(x, w, b, conv_param, gamma, beta, bn_param, pool_param):
  """
  Convenience layer that performs a convolution, batchnorm, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  bn, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(bn)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache

def conv_bn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  dbn = relu_backward(ds, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(dbn, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta

def conv_bn_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
  """
  Convenience layer that performs a convolution, batchnorm, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  bn, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  cache = (conv_cache, bn_cache, relu_cache)
  return out, cache

def conv_bn_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, bn_cache, relu_cache = cache
  dbn = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(dbn, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta




class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv (-batch_norm) - relu - 2x2 max pool - affine (-bactch_norm) - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               use_batchnorm=False, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ###################################### ######################################
    
    # XH: arbitrarily set a defaut value for stride and padding of ConvNet and MaxPool
    Sconv = 1
    pad = (filter_size - 1)/2
    Hp = 2
    Wp = 2
    Sp = 2    
    # end of arbitrarily add default parameters
    
    C, H, W = input_dim    
    Hf = filter_size
    Wf = filter_size
    HO = (H +2*pad - Hf)/Sconv +1
    WO = (W +2*pad - Wf)/Sconv +1
    HO = (HO - Hp)/Sp + 1
    WO = (WO - Wp)/Sp +1
    D = num_filters * HO * WO
    
    W1 = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
    W2 = np.random.randn(D, hidden_dim) * weight_scale
    W3 = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['W1'] = W1
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = W2
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = W3
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    if self.use_batchnorm:
        self.bn_params = [{'mode':'train'} for i in xrange(2)] #we only need two bn layers here
        # note here the input for batch_norm is of shape (N,F,HO,WO)        
        self.params['gamma1'] = np.ones(num_filters)
        self.params['beta1'] = np.zeros(num_filters)
        self.params['gamma2'] = np.ones(hidden_dim)
        self.params['beta2'] = np.zeros(hidden_dim)
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode
        
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if self.use_batchnorm:
        bn_param1 = self.bn_params[0]
        gamma1 = self.params['gamma1']
        beta1 = self.params['beta1']
        conv_out, conv_cache = conv_bn_relu_pool_forward(X, W1, b1, conv_param, gamma1, beta1, bn_param1, pool_param)
        N = X.shape[0]    
        out = np.reshape(conv_out, (N, -1))
        bn_param2 = self.bn_params[1]
        gamma2 = self.params['gamma2']
        beta2 = self.params['beta2']
        out, aff1_cache = affine_bn_relu_forward(out, W2, b2, gamma2, beta2, bn_param2)
        scores, aff2_cache = affine_forward(out, W3, b3)
    else:
        conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        N = X.shape[0]    
        out = np.reshape(conv_out, (N, -1))
        out, aff1_cache = affine_relu_forward(out, W2, b2)
        scores, aff2_cache = affine_forward(out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dloss = softmax_loss(scores, y)
    reg = self.reg
    loss += 0.5*reg * (np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
    
    #compute gradients
    if self.use_batchnorm:
        daff2, dw, db = affine_backward(dloss, aff2_cache)
        grads['W3'] = dw + reg*W3
        grads['b3'] = db
        daff1, dw, db, dgamma, dbeta = affine_bn_relu_backward(daff2, aff1_cache)
        grads['W2'] = dw + reg*W2
        grads['b2'] = db
        grads['gamma2'] = dgamma
        grads['beta2'] = dbeta
        daff1 = np.reshape(daff1, conv_out.shape)
        dx, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(daff1, conv_cache)
        grads['W1'] = dw + reg*W1
        grads['b1'] = db
        grads['gamma1'] = dgamma
        grads['beta1'] = dbeta
    else:
        daff2, dw, db = affine_backward(dloss, aff2_cache)
        grads['W3'] = dw + reg*W3
        grads['b3'] = db
        daff1, dw, db = affine_relu_backward(daff2, aff1_cache)
        grads['W2'] = dw + reg*W2
        grads['b2'] = db
        daff1 = np.reshape(daff1, conv_out.shape)
        dx, dw, db = conv_relu_pool_backward(daff1, conv_cache)
        grads['W1'] = dw + reg*W1
        grads['b1'] = db
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
class C3A2Net(object):
  """
  
  (conv-batch_norm-relu)X3 - 2x2 max pool - (affine-bactch_norm-relu)X2 - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ###################################### ######################################
    
    # XH: arbitrarily set a defaut value for stride and padding of ConvNet and MaxPool
    Sconv = 1
    pad = (filter_size - 1)/2
    Hp = 2
    Wp = 2
    Sp = 2    
    # end of arbitrarily add default parameters
    
    C, H, W = input_dim    
    Hf = filter_size
    Wf = filter_size
    H_1 = (H +2*pad - Hf)/Sconv +1
    W_1 = (W +2*pad - Wf)/Sconv +1
    H_2 = (H_1 +2*pad - Hf)/Sconv +1
    W_2 = (W_1 +2*pad - Wf)/Sconv +1
    H_3 = (H_2 +2*pad - Hf)/Sconv +1
    W_3 = (W_2 +2*pad - Wf)/Sconv +1
    HO = (H_3 - Hp)/Sp + 1
    WO = (W_3 - Wp)/Sp +1
    # output of convnetX3 is (N, F, HO, WO)
    D = num_filters * HO * WO
    
    W1 = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
    W2 = np.random.randn(num_filters, num_filters, filter_size, filter_size) * weight_scale
    W3 = np.random.randn(num_filters, num_filters, filter_size, filter_size) * weight_scale
    W4 = np.random.randn(D, hidden_dim) * weight_scale
    W5 = np.random.randn(hidden_dim, hidden_dim) * weight_scale
    W6 = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['W1'] = W1
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = W2
    self.params['b2'] = np.zeros(num_filters)
    self.params['W3'] = W3
    self.params['b3'] = np.zeros(num_filters)
    self.params['W4'] = W4
    self.params['b4'] = np.zeros(hidden_dim)
    self.params['W5'] = W5
    self.params['b5'] = np.zeros(hidden_dim)
    self.params['W6'] = W6
    self.params['b6'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    self.bn_params = [{'mode':'train'} for i in xrange(5)] #we only need two bn layers here
    # note here the input for batch_norm is of shape (N,F,HO,WO)        
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    self.params['gamma2'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    self.params['gamma3'] = np.ones(num_filters)
    self.params['beta3'] = np.zeros(num_filters)
    self.params['gamma4'] = np.ones(hidden_dim)
    self.params['beta4'] = np.zeros(hidden_dim)
    self.params['gamma5'] = np.ones(hidden_dim)
    self.params['beta5'] = np.zeros(hidden_dim)
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    for bn_param in self.bn_params:
      bn_param['mode'] = mode
        
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    bn_param1 = self.bn_params[0]
    gamma1 = self.params['gamma1']
    beta1 = self.params['beta1']
    conv_out1, conv_cache1 = conv_bn_relu_forward(X, W1, b1, conv_param, gamma1, beta1, bn_param1)
    bn_param2 = self.bn_params[1]
    gamma2 = self.params['gamma2']
    beta2 = self.params['beta2']
    conv_out2, conv_cache2 = conv_bn_relu_forward(conv_out1, W2, b2, conv_param, gamma2, beta2, bn_param2)
    bn_param3 = self.bn_params[2]
    gamma3 = self.params['gamma3']
    beta3 = self.params['beta3']
    conv_out3, conv_cache3 = conv_bn_relu_pool_forward(conv_out2, W3, b3, conv_param, gamma3, beta3, bn_param3, pool_param)
    
    N = X.shape[0]    
    out = np.reshape(conv_out3, (N, -1))
    bn_param4 = self.bn_params[3]
    gamma4 = self.params['gamma4']
    beta4 = self.params['beta4']
    affout1, aff1_cache = affine_bn_relu_forward(out, W4, b4, gamma4, beta4, bn_param4)
    bn_param5 = self.bn_params[4]
    gamma5 = self.params['gamma5']
    beta5 = self.params['beta5']
    affout2, aff2_cache = affine_bn_relu_forward(affout1, W5, b5, gamma5, beta5, bn_param5)
    scores, aff3_cache = affine_forward(affout2, W6, b6)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dloss = softmax_loss(scores, y)
    reg = self.reg
    loss += 0.5*reg * (np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)+np.sum(W4*W4)+np.sum(W5*W5)+np.sum(W6*W6))
    
    #compute gradients
    daff3, dw, db = affine_backward(dloss, aff3_cache)
    grads['W6'] = dw + reg*W6
    grads['b6'] = db
    daff2, dw, db, dgamma, dbeta = affine_bn_relu_backward(daff3, aff2_cache)
    grads['W5'] = dw + reg*W5
    grads['b5'] = db
    grads['gamma5'] = dgamma
    grads['beta5'] = dbeta
    daff1, dw, db, dgamma, dbeta = affine_bn_relu_backward(daff2, aff1_cache)
    grads['W4'] = dw + reg*W4
    grads['b4'] = db
    grads['gamma4'] = dgamma
    grads['beta4'] = dbeta
    
    daff1 = np.reshape(daff1, conv_out3.shape)

    dconv2, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(daff1, conv_cache3)
    grads['W3'] = dw + reg*W3
    grads['b3'] = db
    grads['gamma3'] = dgamma
    grads['beta3'] = dbeta
    dconv1, dw, db, dgamma, dbeta = conv_bn_relu_backward(dconv2, conv_cache2)
    grads['W2'] = dw + reg*W2
    grads['b2'] = db
    grads['gamma2'] = dgamma
    grads['beta2'] = dbeta
    dx, dw, db, dgamma, dbeta = conv_bn_relu_backward(dconv1, conv_cache1)
    grads['W1'] = dw + reg*W1
    grads['b1'] = db
    grads['gamma1'] = dgamma
    grads['beta1'] = dbeta
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads