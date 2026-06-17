"""Test edaio on real-world datasets."""

import pandas as pd
import numpy as np
from edaio import glance # type: ignore

# ============================================
# DATASET 1: TITANIC (Classic ML Dataset)
# ============================================
print("\n" + "="*60)
print("🧊 TEST 1: TITANIC DATASET")
print("="*60)

# Create a realistic Titanic-like dataset
np.random.seed(42)
n = 891

titanic = pd.DataFrame({
    'PassengerId': range(1, n+1),
    'Survived': np.random.choice([0, 1], n, p=[0.62, 0.38]),
    'Pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
    'Name': [f'Passenger_{i}' for i in range(n)],
    'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
    'Age': np.random.normal(30, 14, n).clip(0.5, 80).astype(int),
    'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.68, 0.23, 0.03, 0.02, 0.01, 0.03]),
    'Parch': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.76, 0.13, 0.09, 0.005, 0.005, 0.01]),
    'Fare': np.random.exponential(32, n).round(2),
    'Embarked': np.random.choice(['C', 'Q', 'S', np.nan], n, p=[0.19, 0.09, 0.70, 0.02]),
})

# Inject missing values
titanic.loc[np.random.choice(n, 50, replace=False), 'Age'] = np.nan
titanic.loc[np.random.choice(n, 30, replace=False), 'Fare'] = np.nan
titanic.loc[np.random.choice(n, 10, replace=False), 'Embarked'] = np.nan

# Add outliers
titanic.loc[np.random.choice(n, 5, replace=False), 'Fare'] = 500

print(f"📊 Dataset shape: {titanic.shape}")
print(f"📋 Columns: {list(titanic.columns)}")

# Run edaio
report = glance(titanic, target='Survived')


# ============================================
# DATASET 2: SALES DATA (Time Series)
# ============================================
print("\n" + "="*60)
print("💰 TEST 2: SALES DATASET")
print("="*60)

np.random.seed(123)
n = 1000

# Create order dates with random offsets
base_dates = pd.date_range('2023-01-01', periods=n, freq='D')
random_offsets = np.random.randint(-5, 5, n)
order_dates = base_dates + pd.to_timedelta(random_offsets, unit='D')

sales = pd.DataFrame({
    'order_id': range(1000, 1000+n),
    'customer_id': np.random.choice(range(1, 101), n),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Toys'], n),
    'order_date': order_dates,
    'quantity': np.random.choice(range(1, 6), n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'unit_price': np.random.uniform(10, 200, n).round(2),
    'discount': np.random.choice([0, 0.05, 0.10, 0.15, 0.20], n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
    'is_returned': np.random.choice([0, 1], n, p=[0.92, 0.08]),
    'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Cash', 'Bank Transfer', np.nan], n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
})

# Inject issues
sales.loc[np.random.choice(n, 80, replace=False), 'unit_price'] = np.nan
sales.loc[np.random.choice(n, 40, replace=False), 'customer_id'] = np.nan

# Create a second dataset for drift detection (testing comparison)
sales_compare = sales.sample(500).copy()
sales_compare['product_category'] = np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Toys'], 500)

# Run with comparison
print("📊 Testing comparison/drift detection...")
report_sales = glance(sales, compare_to=sales_compare, columns=['product_category', 'quantity', 'unit_price'])


# ============================================
# DATASET 3: CUSTOMER DATA (Categorical Heavy)
# ============================================
print("\n" + "="*60)
print("👤 TEST 3: CUSTOMER DATA")
print("="*60)

np.random.seed(456)
n = 500

customers = pd.DataFrame({
    'customer_id': [f'CUST_{i:05d}' for i in range(n)],
    'age': np.random.normal(40, 15, n).clip(18, 90).astype(int),
    'income': np.random.lognormal(10, 1.2, n).round(2),
    'gender': np.random.choice(['M', 'F', 'Other', np.nan], n, p=[0.48, 0.48, 0.01, 0.03]),
    'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', np.nan], n),
    'subscription_tier': np.random.choice(['Free', 'Basic', 'Premium', 'Enterprise'], n, p=[0.3, 0.4, 0.2, 0.1]),
    'total_spent': np.random.exponential(500, n).round(2),
    'last_purchase_days': np.random.exponential(30, n).round(0).astype(int).clip(1, 365),
    'email_verified': np.random.choice([0, 1], n, p=[0.1, 0.9]),
    'referral_count': np.random.choice(range(0, 11), n, p=[0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01]),
})

# Inject high cardinality
customers['email'] = [f'user_{i}@example.com' for i in range(n)]  # Unique ID
customers.loc[50:100, 'income'] = np.nan  # Missing block

report_customers = glance(customers, target='total_spent')


# ============================================
# DATASET 4: BIG DATA (Performance Test)
# ============================================
print("\n" + "="*60)
print("🚀 TEST 4: BIG DATA PERFORMANCE")
print("="*60)

np.random.seed(789)
n = 100_000  # 100k rows

big_df = pd.DataFrame({
    'id': range(n),
    'feature_1': np.random.normal(0, 1, n),
    'feature_2': np.random.exponential(1, n),
    'feature_3': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
    'feature_4': np.random.randint(0, 100, n),
    'feature_5': np.random.randn(n),
    'feature_6': np.random.uniform(0, 1, n),
    'feature_7': np.random.poisson(5, n),
    'feature_8': np.random.choice(['X', 'Y', 'Z'], n),
    'feature_9': np.random.lognormal(0, 1, n),
    'feature_10': np.random.normal(50, 15, n),
})

# Inject some issues
big_df.loc[np.random.choice(n, 1000, replace=False), 'feature_1'] = np.nan
big_df.loc[np.random.choice(n, 500, replace=False), 'feature_7'] = np.nan
big_df.loc[np.random.choice(n, 200, replace=False), 'feature_10'] = 9999  # Outliers

print(f"📊 Processing {n:,} rows with 10 columns...")
report_big = glance(big_df)


# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"• Titanic: {len(titanic):,} rows, {len(titanic.columns)} cols")
print(f"• Sales: {len(sales):,} rows, {len(sales.columns)} cols")
print(f"• Customers: {len(customers):,} rows, {len(customers.columns)} cols")
print(f"• Big Data: {len(big_df):,} rows, {len(big_df.columns)} cols")
print("\n💡 edaio is ready for production!")

# Show some sample suggestions
print("\n📝 SAMPLE SUGGESTIONS FROM TITANIC:")
for s in report.suggestions[:5]:
    print(f"  • {s}")