import torch


def lrt_flip_scheme(scores_bar, targets, delta):
    """


    The LRT correction scheme.
    scores_bar (softlabels) is the prediction of the network which is compared with
      noisy label y_tilde.
    If the LR is smaller than the given threshhold delta, we reject LRT and flip y_tilde
      to prediction of scores_bar

    Input
    scores_bar: rolling average of output after softlayers for past 10 epochs.
      Could use other rolling windows.
    y_tilde: noisy labels at current epoch
    delta: LRT threshholding

    Output
    new_y_tilde : new noisy labels after cleanning
    clean_softlabels : softversion of y_tilde


    << Extracted from "Error-Bounded Correction of Noisy Labels
    Songzhu Zheng, Pengxiang Wu, Aman Goswami, Mayank Goswami, Dimitris Metaxas,
      Chao Chen
    Paper Link Presented at ICML 2020" >>
    """
    if targets.shape[1] > 1:
        y_tilde = targets.argmax(dim=1)
    else:
        y_tilde = targets.squeeze()

    cond_1 = torch.logical_not(scores_bar.argmax(1) == y_tilde)
    pred_softlabels_bar_max, _ = scores_bar.max(1)
    ratio = (
        pred_softlabels_bar_max
        / (scores_bar[torch.arange(scores_bar.size(0)), y_tilde])
    )
    print(ratio.mean(), ratio.std(), ratio.min(), ratio.max())
    cond_2 = ratio > delta

    condition = torch.logical_and(cond_1, cond_2)
    new_y_tilde = torch.where(condition, scores_bar.argmax(1), y_tilde)

    changed_idx = torch.where(new_y_tilde != y_tilde)[0]

    # eps = 1e-2
    # clean_softlabels = torch.ones(ntrain, num_class) * eps / (num_class - 1)
    # clean_softlabels.scatter_(
    #     1, torch.tensor(np.array(y_tilde)).reshape(-1, 1), 1 - eps
    # )
    return new_y_tilde, changed_idx
