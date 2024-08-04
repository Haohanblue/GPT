from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
# 创建一个包含预测标签和实际标签的列表
actual_labels = [-1, -1, 0, -1, 0, 0, 0, 1, 1, 0, 1, -1, 1, 0, 0, 1, -1, -1, -1, 0, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 0, 1, 0, -1, 1, 1, 1, 0, -1, -1, 1, -1, 1, 1, 1, 1, 0, 1, 1, 1, -1, -1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0]

# 计算准确率
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels, average='weighted')
recall = recall_score(actual_labels, predicted_labels, average='weighted')
f1 = f1_score(actual_labels, predicted_labels, average='weighted')

# 打印结果
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
