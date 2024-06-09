import pandas as pd
from faker import Faker
import random

fake = Faker()

def generate_brands(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'Brand_ID': fake.uuid4(),
            'Brand_Name': fake.company(),
            'Brand_Description': fake.catch_phrase()
        })
    return pd.DataFrame(data)

def generate_products(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'Product_ID': fake.uuid4(),
            'Product_Name': fake.word(),
            'Product_Price': round(random.uniform(10.0, 1000.0), 2),
            'Brand_ID': fake.uuid4()  # This should match Brand_ID from generate_brands
        })
    return pd.DataFrame(data)

def generate_customers(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'Customer_ID': fake.uuid4(),
            'Name': fake.name(),
            'Email': fake.email(),
            'Phone': fake.phone_number()
        })
    return pd.DataFrame(data)

def generate_orders(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'Order_Num': fake.uuid4(),
            'Customer_ID': fake.uuid4(),  # This should match Customer_ID from generate_customers
            'Order_Date': fake.date_time_this_year()
        })
    return pd.DataFrame(data)

def generate_order_details(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'Order_Num': fake.uuid4(),  # This should match Order_Num from generate_orders
            'Product_ID': fake.uuid4(),  # This should match Product_ID from generate_products
            'Quantity': random.randint(1, 10),
            'Price': round(random.uniform(10.0, 1000.0), 2)
        })
    return pd.DataFrame(data)

def generate_employees(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'Employee_ID': fake.uuid4(),
            'Name': fake.name(),
            'Position': fake.job(),
            'Branch_ID': fake.uuid4()  # This should match Branch_ID from generate_branches
        })
    return pd.DataFrame(data)

def generate_branches(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'Branch_ID': fake.uuid4(),
            'Branch_Name': fake.city(),
            'Location': fake.address()
        })
    return pd.DataFrame(data)

def generate_suppliers(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'Supplier_ID': fake.uuid4(),
            'Supplier_Name': fake.company(),
            'Contact_Name': fake.name(),
            'Phone': fake.phone_number()
        })
    return pd.DataFrame(data)

def generate_deliveries(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'Delivery_ID': fake.uuid4(),
            'Order_Num': fake.uuid4(),  # This should match Order_Num from generate_orders
            'Delivery_Date': fake.date_time_this_year(),
            'Carrier_Name': fake.company()
        })
    return pd.DataFrame(data)

def generate_shipping(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'Shipping_ID': fake.uuid4(),
            'Order_Num': fake.uuid4(),  # This should match Order_Num from generate_orders
            'Shipping_Method': fake.bs(),
            'Shipping_Cost': round(random.uniform(5.0, 50.0), 2)
        })
    return pd.DataFrame(data)

# Generate datasets
num_entries = 200

brands = generate_brands(num_entries)
products = generate_products(num_entries)
customers = generate_customers(num_entries)
orders = generate_orders(num_entries)
order_details = generate_order_details(num_entries)
employees = generate_employees(num_entries)
branches = generate_branches(num_entries)
suppliers = generate_suppliers(num_entries)
deliveries = generate_deliveries(num_entries)
shipping = generate_shipping(num_entries)

# Save to CSV
brands.to_csv('sample_brands.csv', index=False)
products.to_csv('sample_products.csv', index=False)
customers.to_csv('sample_customers.csv', index=False)
orders.to_csv('sample_orders.csv', index=False)
order_details.to_csv('sample_order_details.csv', index=False)
employees.to_csv('sample_employees.csv', index=False)
branches.to_csv('sample_branches.csv', index=False)
suppliers.to_csv('sample_suppliers.csv', index=False)
deliveries.to_csv('sample_deliveries.csv', index=False)
shipping.to_csv('sample_shipping.csv', index=False)
