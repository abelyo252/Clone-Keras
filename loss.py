"""
Loss Function Module
By: Abel Yohannes
Website: https://github.com/abelyo252/
"""

import numpy as np


class Loss:

    def forward(self,y_pred ,y_true):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self,y_pred ,y_true):
        raise NotImplementedError("Backward method not implemented.")



class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true,training=True):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded, convert them to discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy dvalues
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

        # Return gradient
        #print("Dinput Caterogical cross",self.dinputs)
        return self.dinputs


# Custom loss function example
class CustomLoss(Loss):
    def forward(self):

        raise NotImplementedError("Forward method not implemented.")

    def backward(self):
        # Calculate gradients for custom loss function
        raise NotImplementedError("Backward method not implemented.")

# See Github or Telegram Address for help at https://github.com/abelyo252/
# https://t.me/benyohanan