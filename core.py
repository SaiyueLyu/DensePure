from math import ceil

import numpy as np
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
import torch

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, budget_jump_to_guiding_ratio: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        # added by Saiyue for our method
        self.budget_jump_to_guiding_ratio = budget_jump_to_guiding_ratio
        # sigma_guide = ratio * sigma_jump
        self.jump_sigma = self.sigma * np.sqrt(1 + 1 / self.budget_jump_to_guiding_ratio ** 2) if self.budget_jump_to_guiding_ratio != 0 else self.sigma
        self.guide_sigma = self.sigma * np.sqrt(1 + self.budget_jump_to_guiding_ratio ** 2)
        print(f"jump sigma is {self.jump_sigma}, guide sigma is {self.guide_sigma}")
        print(np.sqrt(1/ (1 / self.jump_sigma ** 2 + 1/ self.guide_sigma ** 2)))
        print(f"adding noise with sigma {self.jump_sigma}")

    def certify(self, x: torch.tensor, n0: int, n: int, sample_id:int, alpha: float, batch_size: int, clustering_method='none') -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection, n0_predictions = self._sample_noise(x, n0, batch_size, clustering_method, sample_id)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation, n_predictions = self._sample_noise(x, n, batch_size, clustering_method, sample_id)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0, n0_predictions, n_predictions
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius, n0_predictions, n_predictions

    def certify_noapproximate(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        top2 = counts_estimation.argsort()[::-1][:2]
        nA = counts_estimation[top2[0]].item()
        nB = counts_estimation[top2[1]].item()

        pABar = self._lower_confidence_bound(nA, n, alpha)
        pBBar = self._upper_confidence_bound(nB, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma/2 * (norm.ppf(pABar) - norm.ppf(pBBar))
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size, clustering_method='none', sample_id=None, original_x=None) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            predictions_all = np.array([], dtype=int)
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.jump_sigma
                # print(batch.shape)
                # print(f"self sigma is {self.sigma}")
                # print(f" batch range {batch.min()}")

                if clustering_method == 'classifier':
                    predictions = self.base_classifier(batch + noise, sample_id, batch).argmax(1)
                    predictions = predictions.view(this_batch_size,-1).cpu().numpy()
                    count_max_list = np.zeros(this_batch_size,dtype=int)
                    for i in range(this_batch_size):
                        count_max = max(list(predictions[i]),key=list(predictions[i]).count)
                        count_max_list[i] = count_max
                    counts += self._count_arr(count_max_list, self.num_classes)

                else:
                    # print(f"noise is {(noise).min():.3f}, {(noise).max():.3f}")
                    # print(f"img is {(batch).min():.3f}, {(batch).max():.3f}")
                    # print(f"noisy img is {(batch + noise).min():.3f}, {(batch + noise).max():.3f}")  # [0,1]
                    predictions = self.base_classifier(batch + noise, sample_id, batch).argmax(1)
                    counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
                    predictions_all = np.hstack((predictions_all, predictions.cpu().numpy()))
            
            return counts, predictions_all

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def _upper_confidence_bound(self, NA: int, N: int, alpha: float) -> float:

        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[1]
