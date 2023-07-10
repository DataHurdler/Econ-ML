import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# Step 1: Calculate the average value for each model and type
averages = df.groupby(['model', 'n_test'])['mape'].mean().reset_index()

fig, ax = plt.subplots(figsize=(15, 5))
# Step 2: Create a line plot for each model
models = averages['model'].unique()
for model in models:
    # Determine the line style based on the model name
    linestyle = '--o' if 'single' in model else '-o'

    model_data = averages[averages['model'] == model]
    ax.plot(model_data['n_test'], model_data['mape'], linestyle, label=model)

    # Add model name as a text label on the line
    # last_index = len(model_data) - 1
    # x = model_data['n_test'].iloc[last_index]
    # y = model_data['mape'].iloc[last_index]
    # plt.text(x, y, model, ha='left')

# Set plot title and labels
ax.set_title('Average Value by Type for Each Model')
ax.set_xlabel('Forecast Horizon')
ax.set_ylabel('Mean Absolute Percentage Error')

# Show a legend indicating the models
plt.legend()

# Save the plot
plt.savefig("comparison_dl.png", dpi=300)
