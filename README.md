# Customer Segmentation using K-Means Clustering

An unsupervised machine learning project that segments mall customers into distinct groups based on their annual income and spending patterns.

## 📊 Dataset
- **Source**: [Mall Customer Segmentation (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Features**: Customer ID, gender, age, annual income, spending score
- **Clustering Features**: Annual income, spending score

## 🛠️ Technologies Used
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 📁 Project Structure
```
├── main.py                           # Main script
├── Mall_Customers.csv                # Dataset (download separately)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/ML-Task2-Customer-Segmentation.git
cd ML-Task2-Customer-Segmentation
```

2. **Create a virtual environment:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
- Download from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Save as `Mall_Customers.csv` in the project folder

5. **Run the script:**
```bash
python main.py
```

## 📈 Features
- Data exploration and visualization
- Feature scaling using StandardScaler
- Elbow method to determine optimal number of clusters
- K-Means clustering implementation
- Cluster visualization and analysis
- Customer group profiling
- **Bonus**: DBSCAN clustering comparison

## 📊 Results
The model successfully segments customers into 5 distinct groups:
- High income, high spending
- High income, low spending
- Medium income, medium spending
- Low income, high spending
- Low income, low spending

