import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from scipy import stats

def perform_multiple_regression(excel_path='data.xlsx',
                              feature_columns=['Feature1', 'Feature2', 'Feature3'],
                              target_column='Target',
                              test_size=0.2,
                              random_state=42,
                              output_prefix='multiple_regression'):
    """
    Performs multiple regression analysis with comprehensive diagnostics.
    
    Parameters:
    -----------
    excel_path : str
        Path to Excel file containing the data
    feature_columns : list
        List of column names to use as predictors
    target_column : str
        Name of the column containing the target variable
    test_size : float
        Proportion of data to use for testing (0.0 to 1.0)
    random_state : int
        Random seed for reproducibility
    output_prefix : str
        Prefix for output files
    """
    
    # Read the data
    print(f"Reading data from {excel_path}...")
    df = pd.read_excel(excel_path)
    
    # Prepare features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Fit sklearn model for predictions
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Fit statsmodels for detailed statistics
    X_train_sm = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Create coefficient summary
    coef_summary = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_,
        'Std Error': model_sm.bse[1:],
        't-value': model_sm.tvalues[1:],
        'p-value': model_sm.pvalues[1:],
        'Standardized Coefficient': model.coef_ * np.std(X[feature_columns], axis=0)
    })
    
    # Calculate VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    
    # Create results dictionary
    results = {
        'R-squared': r2,
        'Adjusted R-squared': adj_r2,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae,
        'F-statistic': model_sm.fvalue,
        'F-statistic p-value': model_sm.f_pvalue,
        'Number of Observations': len(y),
        'Number of Features': len(feature_columns)
    }
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save results to Excel
    excel_output = f'{output_prefix}_results_{timestamp}.xlsx'
    with pd.ExcelWriter(excel_output) as writer:
        pd.DataFrame([results]).to_excel(
            writer, sheet_name='Model Summary', index=False
        )
        coef_summary.to_excel(
            writer, sheet_name='Coefficients', index=False
        )
        vif_data.to_excel(
            writer, sheet_name='VIF Analysis', index=False
        )
        
        # Save test set predictions
        pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Residual': y_test - y_pred
        }).to_excel(writer, sheet_name='Predictions', index=False)
    
    print("\nResults saved to:", excel_output)
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted Plot
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Actual vs Predicted')
    
    # 2. Residual Plot
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    
    # 3. Coefficient Plot
    coef_summary.plot(kind='bar', x='Feature', y='Standardized Coefficient', 
                     ax=ax3)
    ax3.set_title('Standardized Coefficients')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    
    # 4. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    
    # Save plot
    plot_output = f'{output_prefix}_plot_{timestamp}.png'
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to:", plot_output)
    
    # Print results to terminal
    print("\nMultiple Regression Results:")
    print("--------------------------")
    print(f"R-squared: {r2:.4f}")
    print(f"Adjusted R-squared: {adj_r2:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print("\nModel Summary:")
    print(model_sm.summary())
    print("\nVariance Inflation Factors:")
    print(vif_data)

if __name__ == "__main__":
    # Example usage:
    # Modify these parameters according to your data
    perform_multiple_regression(
        excel_path='data.xlsx',
        feature_columns=['Feature1', 'Feature2', 'Feature3'],
        target_column='Target',
        test_size=0.2,
        random_state=42,
        output_prefix='multiple_regression'
    )