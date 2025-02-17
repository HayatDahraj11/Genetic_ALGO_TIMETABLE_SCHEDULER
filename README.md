# Classroom Seating Optimizer

## 📌 Overview
This project is a **Classroom Seating Optimizer** that uses **Genetic Algorithms (GA)** and **Simulated Annealing (SA)** to find an optimal seating arrangement for students in a classroom. The optimization process aims to improve seating arrangements based on predefined constraints, such as front-row preference and spacing between students.

It features a **Tkinter GUI** that allows users to input the number of students, rows, and columns and visualize the optimized seating layout.

## 🚀 Features
- **Genetic Algorithm (GA) & Simulated Annealing (SA):** Hybrid approach for optimal seating.
- **Caching for Performance:** Uses `lru_cache` and `OrderedDict` to speed up calculations.
- **Tkinter GUI:** User-friendly interface for input and visualization.
- **Error Handling & Logging:** Logs errors and warnings for debugging.
- **Progress Bar:** Displays the optimization process in real-time.

---

## 📦 Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/HayatDahraj11/Genetic_ALGO_TIMETABLE_SCHEDULER.git
cd Genetic_ALGO_TIMETABLE_SCHEDULER
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
python main.py
```

---

## 🎮 Usage
1. **Enter the number of students, rows, and columns** in the input fields.
2. Click **"Optimize"** to start the seating optimization process.
3. A **progress window** will appear showing the optimization status.
4. After completion, the **best seating arrangement** will be displayed.

---

## 🛠 Project Structure
```
📂 Genetic_ALGO_TIMETABLE_SCHEDULER
│── main.py            # Tkinter GUI for user interaction
│── genetic_algo.py    # Genetic Algorithm & Simulated Annealing logic
│── README.md          # Documentation
│── requirements.txt   # Dependencies
```

---

## 🧬 Algorithm Details
### **Genetic Algorithm (GA) Workflow**
1. **Initialize Population**: Generate random seating layouts.
2. **Evaluate Fitness**: Score seating arrangements based on:
   - Front row preference
   - Proper spacing between students
3. **Selection**: Use **tournament selection** to pick parents.
4. **Crossover**: Generate a new layout by mixing parent layouts.
5. **Mutation**: Swap student positions with a small probability.
6. **Repeat for multiple generations**.

### **Simulated Annealing (SA) Optimization**
- Introduces **small random changes** to seating layouts.
- Uses **temperature-based probability** to accept worse solutions early.
- Helps **escape local optima** to find the best solution.

---

## 📊 GUI Features
- **Input Fields:** Enter number of students, rows, and columns.
- **Progress Window:** Shows optimization progress.
- **Seating Grid Visualization:** Displays optimized layout with tooltips for student info.
- **Error Handling:** Alerts if input is invalid.

---

## 💡 Future Improvements
- Add **export functionality** to save optimized seating.
- Integrate **machine learning** to analyze past seating patterns.
- Improve **UI/UX** with additional themes.

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

## 🤝 Contributing
Pull requests are welcome! Please follow these steps:
1. **Fork the repository**.
2. **Create a feature branch** (`git checkout -b feature-name`).
3. **Commit changes** (`git commit -m "Added new feature"`).
4. **Push to your fork** (`git push origin feature-name`).
5. **Submit a pull request**.

---

## 🙌 Acknowledgments
Special thanks to **all contributors** and the **open-source community**!

📧 **For any questions, feel free to contact me.**

