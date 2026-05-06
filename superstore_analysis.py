"""
Superstore Data Analysis Script
================================
A production-level Python script for analyzing Superstore sales data.
Uses Pandas, NumPy, Matplotlib, and Seaborn for data analysis and visualization.

Author: Senior Data Analyst
Date: April 2026
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================
# File paths
INPUT_FILE = 'superstore.csv.csv'
OUTPUT_FILE = 'cleaned_superstore.csv'

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:,.2f}'.format)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_dataset(filepath):
    """
    Load the Superstore dataset from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"✓ Successfully loaded: {filepath}")
            print(f"✓ Encoding used: {encoding}")
            print(f"✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            raise
    
    # If all encodings fail
    raise ValueError(f"Could not load file with any of these encodings: {encodings}")


def inspect_dataset(df):
    """
    Perform initial inspection of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to inspect
    """
    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    
    # Display first few rows
    print("\n📋 First 5 rows:")
    print(df.head())
    
    # Display column information
    print("\n📊 Column Information:")
    print(df.dtypes)
    
    # Display summary statistics
    print("\n📈 Summary Statistics:")
    print(df.describe())


# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to clean
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("\n" + "=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    print("\nMissing values by column:")
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0] if missing.sum() > 0 else "No missing values found!")
    
    # Fill missing values (if any)
    # For numeric columns, fill with median
    # For categorical columns, fill with mode
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  → Filled {col} with median")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  → Filled {col} with mode")
    
    return df


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to clean
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("\n" + "=" * 60)
    print("REMOVING DUPLICATES")
    print("=" * 60)
    
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_count - len(df)
    
    print(f"✓ Initial rows: {initial_count}")
    print(f"✓ Duplicates removed: {duplicates_removed}")
    print(f"✓ Final rows: {len(df)}")
    
    return df


def convert_date_columns(df):
    """
    Convert date columns to proper datetime format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to clean
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("\n" + "=" * 60)
    print("CONVERTING DATE COLUMNS")
    print("=" * 60)
    
    date_columns = ['Order Date', 'Ship Date']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%m/%d/%Y')
            print(f"✓ Converted '{col}' to datetime")
    
    return df


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def create_new_features(df):
    """
    Create new features from existing data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to enhance
        
    Returns:
    --------
    pd.DataFrame
        Enhanced dataframe
    """
    print("\n" + "=" * 60)
    print("CREATING NEW FEATURES")
    print("=" * 60)
    
    # Extract Year and Month from Order Date
    if 'Order Date' in df.columns:
        df['Order Year'] = df['Order Date'].dt.year
        df['Order Month'] = df['Order Date'].dt.month
        df['Order Month Name'] = df['Order Date'].dt.month_name()
        print("✓ Created: Order Year, Order Month, Order Month Name")
    
    # Create Profit Margin column (Profit / Sales * 100)
    # Handle division by zero
    df['Profit Margin (%)'] = np.where(
        df['Sales'] > 0,
        (df['Profit'] / df['Sales']) * 100,
        0
    )
    print("✓ Created: Profit Margin (%)")
    
    # Create a flag for profitable vs unprofitable orders
    df['Is Profitable'] = df['Profit'] > 0
    print("✓ Created: Is Profitable (boolean flag)")
    
    # Create discount category
    df['Discount Category'] = pd.cut(
        df['Discount'],
        bins=[-0.01, 0, 0.1, 0.2, 1.0],
        labels=['No Discount', 'Low (0-10%)', 'Medium (10-20%)', 'High (>20%)']
    )
    print("✓ Created: Discount Category")
    
    return df


# =============================================================================
# EDA FUNCTIONS
# =============================================================================

def calculate_key_metrics(df):
    """
    Calculate key business metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    """
    print("\n" + "=" * 60)
    print("KEY BUSINESS METRICS")
    print("=" * 60)
    
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    avg_profit_margin = df['Profit Margin (%)'].mean()
    
    print(f"\n💰 Total Sales:        ${total_sales:,.2f}")
    print(f"💵 Total Profit:       ${total_profit:,.2f}")
    print(f"📊 Average Profit Margin: {avg_profit_margin:.2f}%")
    
    # Additional metrics
    print(f"\n📦 Total Orders:      {len(df):,}")
    print(f"👥 Unique Customers:  {df['Customer ID'].nunique():,}")
    print(f"🏷️  Unique Products:   {df['Product ID'].nunique():,}")
    
    return {
        'total_sales': total_sales,
        'total_profit': total_profit,
        'avg_profit_margin': avg_profit_margin
    }


def analyze_sales_trends(df):
    """
    Analyze sales trends over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
        
    Returns:
    --------
    pd.DataFrame
        Monthly sales summary
    """
    print("\n" + "=" * 60)
    print("SALES TRENDS OVER TIME")
    print("=" * 60)
    
    # Group by year and month
    monthly_sales = df.groupby(['Order Year', 'Order Month']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    monthly_sales['Year-Month'] = monthly_sales['Order Year'].astype(str) + '-' + \
                                   monthly_sales['Order Month'].astype(str).str.zfill(2)
    
    print("\nMonthly Sales Summary (last 12 months):")
    print(monthly_sales.tail(12).to_string(index=False))
    
    return monthly_sales


def analyze_profit_by_category(df):
    """
    Analyze profit by category and sub-category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    """
    print("\n" + "=" * 60)
    print("PROFIT BY CATEGORY AND REGION")
    print("=" * 60)
    
    # Profit by Category
    category_profit = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
    print("\n📁 Profit by Category:")
    for cat, profit in category_profit.items():
        print(f"   {cat}: ${profit:,.2f}")
    
    # Profit by Region
    region_profit = df.groupby('Region')['Profit'].sum().sort_values(ascending=False)
    print("\n🌍 Profit by Region:")
    for region, profit in region_profit.items():
        print(f"   {region}: ${profit:,.2f}")
    
    return category_profit, region_profit


def analyze_discount_impact(df):
    """
    Analyze the impact of discount on profit.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    """
    print("\n" + "=" * 60)
    print("IMPACT OF DISCOUNT ON PROFIT")
    print("=" * 60)
    
    # Group by discount category
    discount_impact = df.groupby('Discount Category').agg({
        'Sales': ['sum', 'mean'],
        'Profit': ['sum', 'mean'],
        'Quantity': 'sum'
    }).round(2)
    
    print("\n📉 Impact by Discount Category:")
    print(discount_impact)
    
    # Calculate correlation between discount and profit
    correlation = df['Discount'].corr(df['Profit'])
    print(f"\n🔗 Correlation between Discount and Profit: {correlation:.4f}")
    
    return discount_impact


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def setup_plot_style():
    """Set up consistent plot styling."""
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_sales_trend(monthly_sales):
    """
    Plot sales trends over time.
    
    Parameters:
    -----------
    monthly_sales : pd.DataFrame
        Monthly sales data
    """
    print("\n📊 Creating Sales Trend visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot sales and profit on dual axis
    ax.plot(monthly_sales['Year-Month'], monthly_sales['Sales'], 
            marker='o', linewidth=2, color='#2E86AB', label='Sales')
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Sales ($)', color='#2E86AB')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax.tick_params(axis='x', rotation=45)
    
    # Add profit line on secondary axis
    ax2 = ax.twinx()
    ax2.plot(monthly_sales['Year-Month'], monthly_sales['Profit'], 
             marker='s', linewidth=2, color='#28A745', label='Profit')
    ax2.set_ylabel('Profit ($)', color='#28A745')
    ax2.tick_params(axis='y', labelcolor='#28A745')
    
    plt.title('Sales and Profit Trends Over Time', fontsize=14, fontweight='bold')
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('visualization_sales_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualization_sales_trend.png")


def plot_profit_by_category(df):
    """
    Plot profit by category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to visualize
    """
    print("📊 Creating Profit by Category visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Profit by Category (bar chart)
    category_profit = df.groupby('Category')['Profit'].sum().sort_values()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    axes[0].barh(category_profit.index, category_profit.values, color=colors)
    axes[0].set_xlabel('Profit ($)')
    axes[0].set_title('Profit by Category', fontweight='bold')
    for i, v in enumerate(category_profit.values):
        axes[0].text(v + 1000, i, f'${v:,.0f}', va='center')
    
    # Profit by Region (bar chart)
    region_profit = df.groupby('Region')['Profit'].sum().sort_values()
    
    axes[1].barh(region_profit.index, region_profit.values, color=colors)
    axes[1].set_xlabel('Profit ($)')
    axes[1].set_title('Profit by Region', fontweight='bold')
    for i, v in enumerate(region_profit.values):
        axes[1].text(v + 1000, i, f'${v:,.0f}', va='center')
    
    plt.tight_layout()
    plt.savefig('visualization_profit_category_region.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualization_profit_category_region.png")


def plot_discount_impact(df):
    """
    Plot the impact of discount on profit.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to visualize
    """
    print("📊 Creating Discount Impact visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot: Profit distribution by discount category
    discount_order = ['No Discount', 'Low (0-10%)', 'Medium (10-20%)', 'High (>20%)']
    df['Discount Category'] = pd.Categorical(df['Discount Category'], 
                                              categories=discount_order, 
                                              ordered=True)
    
    sns.boxplot(data=df, x='Discount Category', y='Profit', ax=axes[0], palette='RdYlGn')
    axes[0].set_title('Profit Distribution by Discount Level', fontweight='bold')
    axes[0].set_xlabel('Discount Category')
    axes[0].set_ylabel('Profit ($)')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Scatter plot: Discount vs Profit
    axes[1].scatter(df['Discount'], df['Profit'], alpha=0.5, c=df['Profit'], 
                    cmap='RdYlGn', edgecolors='none')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Discount')
    axes[1].set_ylabel('Profit ($)')
    axes[1].set_title('Discount vs Profit (Correlation Analysis)', fontweight='bold')
    
    # Add trend line
    z = np.polyfit(df['Discount'], df['Profit'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Discount'].min(), df['Discount'].max(), 100)
    axes[1].plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend Line')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('visualization_discount_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualization_discount_impact.png")


def plot_category_region_heatmap(df):
    """
    Create a heatmap of profit by category and region.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to visualize
    """
    print("📊 Creating Category-Region Heatmap...")
    
    # Create pivot table
    pivot = df.pivot_table(values='Profit', index='Category', columns='Region', 
                           aggfunc='sum', fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(pivot, annot=True, fmt=',.0f', cmap='RdYlGn', center=0,
                linewidths=0.5, ax=ax)
    ax.set_title('Profit Heatmap: Category × Region', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualization_category_region_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualization_category_region_heatmap.png")


def plot_top_performers(df):
    """
    Plot top performing products and sub-categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to visualize
    """
    print("📊 Creating Top Performers visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top 10 Products by Profit
    top_products = df.groupby('Product Name')['Profit'].sum().nlargest(10)
    axes[0].barh(range(len(top_products)), top_products.values, color='#45B7D1')
    axes[0].set_yticks(range(len(top_products)))
    axes[0].set_yticklabels([p[:40] + '...' if len(p) > 40 else p 
                              for p in top_products.index])
    axes[0].set_xlabel('Profit ($)')
    axes[0].set_title('Top 10 Products by Profit', fontweight='bold')
    axes[0].invert_yaxis()
    
    # Sub-Category Performance
    subcat_profit = df.groupby('Sub-Category')['Profit'].sum().sort_values()
    axes[1].barh(subcat_profit.index, subcat_profit.values, 
                 color=['#FF6B6B' if x < 0 else '#4ECDC4' for x in subcat_profit.values])
    axes[1].set_xlabel('Profit ($)')
    axes[1].set_title('Profit by Sub-Category', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualization_top_performers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualization_top_performers.png")


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_cleaned_data(df, output_path):
    """
    Export the cleaned dataset to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The cleaned dataframe
    output_path : str
        Path for the output file
    """
    print("\n" + "=" * 60)
    print("EXPORTING CLEANED DATA")
    print("=" * 60)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Exported cleaned data to: {output_path}")
    print(f"✓ Final shape: {df.shape[0]} rows × {df.shape[1]} columns")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the complete analysis pipeline.
    """
    print("\n" + "=" * 60)
    print("🚀 SUPERSTORE DATA ANALYSIS")
    print("=" * 60)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load the dataset
    df = load_dataset(INPUT_FILE)
    
    # Step 2: Inspect the dataset
    inspect_dataset(df)
    
    # Step 3: Data Cleaning
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = convert_date_columns(df)
    
    # Step 4: Feature Engineering
    df = create_new_features(df)
    
    # Step 5: Exploratory Data Analysis
    metrics = calculate_key_metrics(df)
    monthly_sales = analyze_sales_trends(df)
    category_profit, region_profit = analyze_profit_by_category(df)
    discount_impact = analyze_discount_impact(df)
    
    # Step 6: Visualizations
    setup_plot_style()
    plot_sales_trend(monthly_sales)
    plot_profit_by_category(df)
    plot_discount_impact(df)
    plot_category_region_heatmap(df)
    plot_top_performers(df)
    
    # Step 7: Export cleaned data
    export_cleaned_data(df, OUTPUT_FILE)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\n📁 Output Files:")
    print(f"   • {OUTPUT_FILE}")
    print("   • visualization_sales_trend.png")
    print("   • visualization_profit_category_region.png")
    print("   • visualization_discount_impact.png")
    print("   • visualization_category_region_heatmap.png")
    print("   • visualization_top_performers.png")


if __name__ == "__main__":
    main()