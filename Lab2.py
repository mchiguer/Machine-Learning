#Computing the Cost function 
import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    N= y.shape[0]
    e = y - (tx @ w)   
    loss =  (1 / 2*N) * (e.T @ e) 
    # ***************************************************
    return loss




#Grid Search
from costs import compute_loss


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    #losses est une matrice 2D contenant la valeur de la loss pour chaque combinaison de  w0 et w1
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)   #transforme cet indice en un couple (L,C) pour retrouver la position dans la matrice.
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Paste your implementation of grid_search
#       here when it is done.
def grid_search(y, tx, w0_grid, w1_grid):
    
    losses = np.zeros((len(w0_grid), len(w1_grid)))
    
    for i, w0 in enumerate(w0_grid): #enumerate te donne deux choses à la fois l'indice et la valeur
        for j, w1 in enumerate(w1_grid):
            w = np.array([w0, w1])
            losses[i, j] = compute_loss(y, tx, w)  
    return get_best_parameters(w0_grid, w1_grid,losses)
# ***************************************************



#Gradient Descent
def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient vector
    # ***************************************************
    N = y.shape[0]
    e = y - (tx @ w)   
    grad = -(tx.T @ e) * (1/N) 
    return grad



def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        gradient_w = compute_gradient(y,tx,w)
        loss = compute_loss(y, tx, w)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma * (gradient_w)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws


#Stochastic Gradient Descent 

#Batches
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

    Number of batches = 9

    Batch size = 7                              Remainder = 3
    v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]



from helpers import batch_iter
from costs import compute_loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation. It's the same as the usual gradient.
    # ***************************************************
    N = y.shape[0]
    e = y - (tx @ w)   
    stoch_grad = -(tx.T @ e) * (1/N) 
    return stoch_grad

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: implement stochastic gradient descent.
        # ***************************************************
        for y_batch, X_batch in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
        
            #on genere un batch a partir duquel on va calculer le gradient
            #batch_iter renvoie deux tableaux, un pour les y et un pour les X
            stoch_gardient = compute_stoch_gradient(y_batch, X_batch, w)
            w = w - gamma * stoch_gradient 
            
        loss = compute_loss(y, tx, w)
        ws.append(w.copy())
        losses.append(loss)
        
        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws


#Subgradient Descent 

def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute subgradient gradient vector for MAE
    # ***************************************************
    N = y.shape[0]
    e = y - tx @ w 
    loss = np.mean(np.abs(e)) 
    grad = - (1/N) * (tx.T @ np.sign(e)) 
    
def subgradient_descent_mae(y, tx, initial_w, max_iters=200, gamma=0.05):
    ws = [initial_w]
    losses = []
    for t in range(max_iters):
        g = compute_subgradient_mae(y, tx, w)
        w -= gamma * g
        
        loss = np.mean(np.abs(y - tx @ w))
        
        ws.append(w.copy())
        losses.append(float(loss))
    return losses, ws