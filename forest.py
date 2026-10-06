# Cross validation Balance Accuracy = 93.43%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from joblib import Parallel, delayed # Used to ensure much faster NewDecisionTree creation

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

def calculate_entropy(y):
    if len(y) == 0:
        return 0.0

    counts = np.unique(y, return_counts = True)[1]

    P = counts / len(y)

    P = P[P > 0] # Ensure log(0) does not occur
    return -np.sum(P * np.log2(P))

def calculate_information_gain(parent_y, left_y, right_y):
    parent_entropy = calculate_entropy(parent_y)
    n_total = len(parent_y)

    weighted_child_entropy = ((len(left_y) * calculate_entropy(left_y)) + (len(right_y) * calculate_entropy(right_y))) / n_total

    return parent_entropy - weighted_child_entropy

def find_best_split(X, y, feature_indices, max_thresholds=100):
  best_gain = -1.0
  best_feature = None
  best_threshold = None

  n_total = len(y)
  if n_total == 0:
    return best_gain, best_feature, best_threshold

  # Encode y locally to 0..C-1
  classes, y_encoded = np.unique(y, return_inverse=True)
  num_classes = len(classes)

  # Parent entropy calculation
  _, parent_counts = np.unique(y_encoded, return_counts=True)
  p_parent = parent_counts / n_total
  parent_entropy = -np.sum(p_parent * np.log2(p_parent))

  for feature_index in feature_indices:
    values = X[:, feature_index]
    unique_vals = np.unique(values)

    if len(unique_vals) <= 1:
      continue

    if len(unique_vals) > max_thresholds:
      thresholds = np.percentile(
          unique_vals, np.linspace(0, 100, max_thresholds)
      )
      thresholds = np.unique(thresholds)
    else:
      thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

    left_masks = values[:, None] <= thresholds[None, :]  # Shape: (N, T)

    left_counts = np.sum(left_masks, axis=0)  # Shape: (T,)
    right_counts = n_total - left_counts

    # Only consider thresholds where BOTH sides have > 0 samples
    valid_mask = (left_counts > 0) & (right_counts > 0)
    if not np.any(valid_mask):
      continue

    y_one_hot = np.eye(num_classes)[y_encoded]
    left_class_counts = left_masks.T @ y_one_hot
    right_class_counts = np.sum(y_one_hot, axis=0)[None, :] - left_class_counts

    # Initialize child entropies to 0.0 for all candidate thresholds
    left_entropy = np.zeros(len(thresholds))
    right_entropy = np.zeros(len(thresholds))

    # Calculate probabilities ONLY for valid split thresholds
    if np.any(valid_mask):
      p_left = (
          left_class_counts[valid_mask] / left_counts[valid_mask, None]
      )  # Safe division!
      p_right = (
          right_class_counts[valid_mask] / right_counts[valid_mask, None]
      )  # Safe division!

      p_left_safe = np.where(p_left > 0, p_left, 1.0)
      p_right_safe = np.where(p_right > 0, p_right, 1.0)

      left_entropy[valid_mask] = -np.sum(p_left * np.log2(p_left_safe), axis=1)
      right_entropy[valid_mask] = -np.sum(
          p_right * np.log2(p_right_safe), axis=1
      )

    weighted_child_entropy = (
        left_counts * left_entropy + right_counts * right_entropy
    ) / n_total
    gains = parent_entropy - weighted_child_entropy

    # Invalidate splits that leave 0 samples on either side
    gains[~valid_mask] = -1.0

    max_idx = np.argmax(gains)
    if gains[max_idx] > best_gain:
      best_gain = gains[max_idx]
      best_feature = feature_index
      best_threshold = thresholds[max_idx]

  return best_gain, best_feature, best_threshold

def build_tree(X, y, current_depth, max_depth, min_samples_split, feature_subsample_size, rng):

    n_samples, n_features = X.shape
    number_of_classes = len(np.unique(y))

    values, counts = np.unique(y, return_counts=True)
    most_common_class = values[np.argmax(counts)]

    if(number_of_classes == 1 or (max_depth is not None and current_depth >= max_depth) or n_samples < min_samples_split):
        return Node(value = most_common_class, isLeaf = True)

    if feature_subsample_size is None or feature_subsample_size > n_features:
        feature_subsample_size = n_features

    feature_indices = rng.choice(
        n_features, size=feature_subsample_size, replace = False
    )

    best_gain, best_feature, best_threshold = find_best_split(X, y, feature_indices)

    if best_gain <= 0 or best_feature is None:
        return Node(value = most_common_class, isLeaf = True)

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_child = build_tree(
        X[left_mask],
        y[left_mask],
        current_depth + 1,
        max_depth,
        min_samples_split,
        feature_subsample_size,
        rng
    )
    
    right_child = build_tree(
        X[right_mask],
        y[right_mask],
        current_depth + 1,
        max_depth,
        min_samples_split,
        feature_subsample_size,
        rng
    )

    return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child, isLeaf=False)
    

        

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, isLeaf=False):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.isLeaf = isLeaf

    def is_leaf_node(self):
        return self.isLeaf

class NewDecisionTree: 
    def __init__(self, max_depth=None, min_samples_split = 2, max_features = None, random_state= None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        rng = np.random.default_rng(self.random_state)

        n_features = X.shape[1]

        # Determine feature subsampling size
        n_features = X.shape[1]
        if self.max_features is None:
            feature_subsample_size = n_features

        elif isinstance(self.max_features, float):
            feature_subsample_size = int(self.max_features * n_features)

        elif self.max_features == "sqrt":
            feature_subsample_size = int(np.sqrt(n_features))
        
        elif self.max_features == "log2":
            feature_subsample_size = int(np.log2(n_features))
    
        else:
            feature_subsample_size = self.max_features

        self.root = build_tree(
            X,
            y,
            current_depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            feature_subsample_size=feature_subsample_size,
            rng=rng,
        )
        return self

    def _predict_sample(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)

        return self._predict_sample(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_sample(x, self.root) for x in X])

def _build_single_tree(args):
  (
      X,
      y,
      max_depth,
      min_samples_split,
      n_features_per_tree,
      bootstrap,
      tree_seed,
  ) = args
  n_samples = X.shape[0]
  rng = np.random.default_rng(tree_seed)

  if bootstrap:
    sample_idx = rng.integers(0, n_samples, size=n_samples)
  else:
    sample_idx = np.arange(n_samples)

  X_boot = X[sample_idx]
  y_boot = y[sample_idx]

  tree = NewDecisionTree(
      max_depth=max_depth,
      min_samples_split=min_samples_split,
      max_features=n_features_per_tree,
      random_state=int(rng.integers(0, 1_000_000)),
  )
  tree.fit(X_boot, y_boot)
  return tree


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

    # Generate random seeds for each individual tree
    tree_seeds = [
        int(s) for s in rng.integers(0, 1_000_000, size=self.n_trees)
    ]

    # Package tasks for parallel worker execution
    tasks = [
        (
            X,
            y,
            self.max_depth,
            self.min_samples_split,
            self.n_features_per_tree,
            self.bootstrap,
            seed,
        )
        for seed in tree_seeds
    ]

    # Run tree fitting in parallel across all available CPU cores
    self.trees = Parallel(n_jobs=-1)(
        delayed(_build_single_tree)(task) for task in tasks
    )

    return self

  def predict(self, X):
    all_preds = np.zeros((len(self.trees), X.shape[0]), dtype=object)
    for i, tree in enumerate(self.trees):
      all_preds[i] = tree.predict(X)

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