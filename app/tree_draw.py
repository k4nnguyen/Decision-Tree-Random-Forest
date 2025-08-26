import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load model và vectorizer
# Thay đổi thành merged_RF_model.pkl và merged_RF_vectorizer.pkl nếu muốn dùng RF, tương tự với DT
model = joblib.load("../models/model/merged_RF_model.pkl")
vectorizer = joblib.load("../models/model/merged_RF_vectorizer.pkl")


# Với RF thì chỉ hiển thị được 1 cây nhỏ trong rừng
tree = model.estimators_[0]  # Bạn có thể đổi [1], [2], ...

# Vẽ cây
plt.figure(figsize=(15, 6))
plot_tree(tree,
          # Nếu là RF thì đổi model thành tree
          max_depth=3,
          feature_names=vectorizer.get_feature_names_out(),
          class_names=["Negative", "Positive"],
          filled=True,
          fontsize=8)
plt.title("Một cây trong Decision Tree/Random Forest")
plt.show()