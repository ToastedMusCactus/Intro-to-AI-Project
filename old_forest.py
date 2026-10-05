# Cross validation Balance Accuracy = 92.30%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

TRAIN_FILE = "dry_bean_train.csv"
TEST_FILE  = "dry_bean_test.csv"

train = pd.read_csv(TRAIN_FILE)
test  = pd.read_csv(TEST_FILE)

print("Train shape:", train.shape)
print("Test  shape:", test.shape)
print("Classes    :", train["Class"].unique())

FEATURE_COLS = [c for c in train.columns if c != "Class"]

X_train = train[FEATURE_COLS].values
y_train = train["Class"].values
X_test  = test[FEATURE_COLS].values


def my_kfold_indices(n_samples, k=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        end = start + size
        folds.append(indices[start:end])
        start = end

    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def cross_validate(model_fn, X, y, k=5, seed=42):
    splits = my_kfold_indices(len(X), k=k, seed=seed)
    accs = []
    for train_idx, val_idx in splits:
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        accs.append(balanced_accuracy_score(y[val_idx], pred))
    return float(np.mean(accs)), accs


print("\n--- Tuning a single DecisionTree with hand-coded CV ---")

single_tree_accs = {}
for depth in [5, 10, 15, 20, None]:
    for min_split in [2, 5, 10]:
        model_fn = lambda d=depth, m=min_split: DecisionTreeClassifier(
            criterion="gini",
            max_depth=d,
            min_samples_split=m,
            random_state=0,
        )
        acc, _ = cross_validate(model_fn, X_train, y_train, k=5)
        single_tree_accs[(depth, min_split)] = acc
        print(f"max_depth={str(depth):<5} min_samples_split={min_split:<3} -> CV bal acc = {acc:.4f}")

best_tree_params = max(single_tree_accs, key=single_tree_accs.get)
print(f"\nBest single-tree params: max_depth={best_tree_params[0]}, "
      f"min_samples_split={best_tree_params[1]}, "
      f"CV bal acc = {single_tree_accs[best_tree_params]:.4f}")


class MyRandomForest:
    def __init__(
        self,
        n_trees=25,
        max_depth=None,
        min_samples_split=2,
        n_features_per_tree=None,
        bootstrap=True,
        seed=42,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_per_tree = n_features_per_tree
        self.bootstrap = bootstrap
        self.seed = seed
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.seed)

        if self.n_features_per_tree is None:
            self.n_features_per_tree = max(1, int(np.sqrt(n_features)))

        self.classes_ = np.unique(y)

        for i in range(self.n_trees):
            if self.bootstrap:
                sample_idx = rng.integers(0, n_samples, size=n_samples)
            else:
                sample_idx = np.arange(n_samples)

            X_boot = X[sample_idx]
            y_boot = y[sample_idx]

            feat_idx = rng.choice(n_features, size=self.n_features_per_tree,
                                  replace=False)

            tree = DecisionTreeClassifier(
                criterion="gini",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees.append((tree, feat_idx))

        return self

    def predict(self, X):
        all_preds = np.zeros((len(self.trees), X.shape[0]), dtype=object)
        for i, (tree, feat_idx) in enumerate(self.trees):
            all_preds[i] = tree.predict(X[:, feat_idx])

        final = []
        for col in range(X.shape[0]):
            votes = all_preds[:, col]
            values, counts = np.unique(votes, return_counts=True)
            final.append(values[np.argmax(counts)])
        return np.array(final)


def forest_model_fn(n_trees, max_depth, min_samples_split,
                    n_features_per_tree, seed):
    return lambda: MyRandomForest(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_features_per_tree=n_features_per_tree,
        bootstrap=True,
        seed=seed,
    )


print("\n--- Tuning MyRandomForest with hand-coded CV ---")

forest_results = {}
for n_trees in [25]:
    for n_feat in [8]:
        for depth in [10, 15]:
            model_fn = forest_model_fn(
                n_trees=n_trees,
                max_depth=depth,
                min_samples_split=2,
                n_features_per_tree=n_feat,
                seed=0,
            )
            acc, _ = cross_validate(model_fn, X_train, y_train, k=5)
            forest_results[(n_trees, n_feat, depth)] = acc
            print(f"n_trees={n_trees:<3} n_feat={n_feat:<2} depth={str(depth):<5} "
                  f"-> CV bal acc = {acc:.4f}")

best_forest_params = max(forest_results, key=forest_results.get)
best_forest_acc = forest_results[best_forest_params]
print(f"\nBest forest params: n_trees={best_forest_params[0]}, "
      f"n_features_per_tree={best_forest_params[1]}, "
      f"max_depth={best_forest_params[2]}, "
      f"CV bal acc = {best_forest_acc:.4f}")


print("\n--- Training final forest on the whole training set ---")

final_forest = MyRandomForest(
    n_trees=25,
    max_depth=best_forest_params[2],
    min_samples_split=2,
    n_features_per_tree=best_forest_params[1],
    bootstrap=True,
    seed=42,
).fit(X_train, y_train)

test_pred = final_forest.predict(X_test)

output = test.copy()
output["Target"] = test_pred
output.to_csv("forest.csv", index=False)

print("\nWrote forest.csv")
print("Preview:")
print(output.head())
print(f"\nCV Balance Accuracy on training data: {best_forest_acc:.4f}")
print(f">>> First line: # Cross validation Balance Accuracy = {best_forest_acc*100:.2f}%")