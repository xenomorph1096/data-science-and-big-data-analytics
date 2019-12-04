from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


bostonds = load_boston()

bdsdf = pd.DataFrame(bostonds['data'], columns=bostonds['feature_names'])
sns.pairplot(bdsdf)
plt.show()

