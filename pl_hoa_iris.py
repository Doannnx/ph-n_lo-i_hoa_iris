
import csv
import pandas as pd
import random
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox

# Đọc dữ liệu từ file CSV
df = pd.read_csv('D:/Phân loại hoa Iris/Phân loại hoa Iris/Iris.csv')

# Chuyển đổi dữ liệu thành danh sách
array = df.values.tolist()

# Trộn dữ liệu ngẫu nhiên
random.shuffle(array)

# Chia dữ liệu thành tập huấn luyện (2/3) `và tập kiểm tra (1/3)
train_size = int(len(array) * 2 / 3)
train_data = array[:train_size]
test_data = array[train_size:]

# Hàm tính xác suất Naive Bayes với hệ số λ
def calculate_probabilities(train_data, lamda=1):
    total_samples = len(train_data)
    class_counts = defaultdict(int)
    feature_counts = defaultdict(lambda: defaultdict(int))

    # Đếm số lượng cho từng lớp và đặc trưng
    for sample in train_data:
        class_label = sample[-1]
        class_counts[class_label] += 1
        for i, feature_value in enumerate(sample[:-1]):
            feature_counts[class_label][(i, feature_value)] += 1

    # Tính xác suất cho từng lớp và đặc trưng với hệ số λ
    probabilities = defaultdict(dict)
    for class_label, count in class_counts.items():
        probabilities[class_label]['prior'] = count / total_samples
        for (i, feature_value), feature_count in feature_counts[class_label].items():
            probabilities[class_label][(i, feature_value)] = (feature_count + lamda) / (count + lamda * len(feature_counts[class_label]))

    return probabilities

# Hàm dự đoán lớp cho một mẫu
def predict(sample, probabilities):
    max_prob = -1
    best_class = None
    for class_label, class_probabilities in probabilities.items():
        prob = class_probabilities['prior']
        for i, feature_value in enumerate(sample[:-1]):
            prob *= class_probabilities.get((i, feature_value), 1 / len(train_data))
        if prob > max_prob:
            max_prob = prob
            best_class = class_label
    return best_class

# Hàm tính độ chính xác của mô hình
def calculate_accuracy(test_data, probabilities):
    correct_predictions = 0
    for sample in test_data:
        predicted_class = predict(sample, probabilities)
        actual_class = sample[-1]
        if predicted_class == actual_class:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_data)
    return accuracy

# Tạo danh sách các giá trị λ để kiểm tra
lambda_values = [0.1, 1, 10]

# Lưu độ chính xác cho từng λ
accuracies = {}

for lamda in lambda_values:
    # Tính toán xác suất với giá trị λ
    probabilities = calculate_probabilities(train_data, lamda)

    # Đánh giá mô hình
    accuracy = calculate_accuracy(test_data, probabilities)
    accuracies[lamda] = accuracy

    print(f"Đo chính xác của mô hình với λ = {lamda}: {accuracy * 100:.2f}%")

# Tìm giá trị λ tốt nhất
best_lambda = max(accuracies, key=accuracies.get)
probabilities = calculate_probabilities(train_data, best_lambda)
best_accuracy = accuracies[best_lambda]

print(f"Giá trị λ tốt nhất là {best_lambda} với độ chính xác {best_accuracy * 100:.2f}%")

# Hàm xử lý sự kiện khi người dùng nhấn nút "Kiểm tra"
def check_flower_type():
    try:
        # Lấy các giá trị đầu vào từ giao diện
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())
        
        # Mẫu đầu vào từ người dùng
        sample = [sepal_length, sepal_width, petal_length, petal_width, None]
        
        # Dự đoán loại hoa
        predicted_class = predict(sample, probabilities)
        
        # Hiển thị kết quả
        result_message = f"Loại hoa dự đoán: {predicted_class}\n"
        
        
        messagebox.showinfo("Kết quả", result_message)
    
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập giá trị hợp lệ cho tất cả các trường!")

# Tạo giao diện người dùng với tkinter
root = tk.Tk()
root.title("Phân loại hoa Iris")

# Thêm nhãn hiển thị độ chính xác và giá trị λ tốt nhất
accuracy_label = tk.Label(root, text=f"Giá trị λ tốt nhất: {best_lambda} - Độ chính xác: {best_accuracy * 100:.2f}%", font=('Arial', 12, 'bold'), fg='blue')
accuracy_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Nhãn và ô nhập liệu cho từng đặc trưng
tk.Label(root, text="Chiều dài lá đài (cm):").grid(row=1, column=0, padx=10, pady=5)
entry_sepal_length = tk.Entry(root)
entry_sepal_length.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Chiều rộng lá đài (cm):").grid(row=2, column=0, padx=10, pady=5)
entry_sepal_width = tk.Entry(root)
entry_sepal_width.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Chiều dài cánh hoa (cm):").grid(row=3, column=0, padx=10, pady=5)
entry_petal_length = tk.Entry(root)
entry_petal_length.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Chiều rộng cánh hoa (cm):").grid(row=4, column=0, padx=10, pady=5)
entry_petal_width = tk.Entry(root)
entry_petal_width.grid(row=4, column=1, padx=10, pady=5)

# Nút kiểm tra
check_button = tk.Button(root, text="Kiểm tra", command=check_flower_type)
check_button.grid(row=5, column=0, columnspan=2, pady=10)

# Khởi chạy giao diện
root.mainloop()
