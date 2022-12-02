import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize
import wandb

# dimensionality reduction
def reduction_features(user_emb):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    user_emb_2d = tsne.fit_transform(user_emb)
    user_emb_2d = normalize(user_emb_2d, axis=1, norm='l2')
    return user_emb_2d


# input: user
def plot_features(embs, name):
    reduction_f = reduction_features(embs)
    f, axs = plt.subplots(2, len(reduction_f), figsize=(12, 3.5),
                          gridspec_kw={'height_ratios': [3, 1]})
    kwargs = {'levels': np.arange(0, 5.5, 0.5)}
    # for i, name in enumerate(models):
    sns.kdeplot(data=reduction_f, bw=0.05, shade=True,
                cmap="GnBu", legend=True, ax=axs[0][0], **kwargs)
    axs[0][0].set_title(name, fontsize=9, fontweight="bold")
    x = [p[0] for p in reduction_f]
    y = [p[1] for p in reduction_f]
    angles = np.arctan2(y, x)
    sns.kdeplot(data=angles, bw=0.15, shade=True,
                legend=True, ax=axs[1][0], color='green')
    artifact = wandb.Artifact('feature', type='dataset')
    sns.savefig(name)
    artifact.add_file(name)
    wandb.log_artifact(artifact)
