import pandas as pd
import openai  # Replace with your LLM interaction library


# Step 1: Load the dataset
def load_dataset(filepath):
    # Example: Read a CSV file into a DataFrame
    df = pd.read_csv(filepath)
    return df


# Step 2: Generate a summary of the data
def generate_data_summary(df):
    summary = {
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.apply(lambda x: x.name).to_dict(),
        "summary_stats": df.describe(include='all').to_dict()
    }
    return summary


# Step 3: Prepare the LLM prompt
def prepare_llm_prompt(summary, user_description=""):
    prompt = f"""
    Below is a summary of a dataset and its quality issues. Please provide tips for improving the data quality, including how to handle missing values, outliers, data consistency, and any other potential improvements. If possible, provide code snippets for implementing these suggestions.

    Data Summary:
    {summary}

    Data Quality Issues:
    {user_description if user_description else "None provided."}
    """
    return prompt


# Step 4: Query the LLM for recommendations
def query_llm(prompt):
    # Replace with your LLM interaction
    response = openai.Completion.create(
        model="gpt-4",  # Example model
        prompt=prompt,
        max_tokens=500
    )
    return response['choices'][0]['text']


# Step 5: Main function to run the script
def main(filepath, user_description=""):
    # Load the dataset
    df = load_dataset(filepath)

    # Generate the data summary
    summary = generate_data_summary(df)

    # Prepare the LLM prompt
    prompt = prepare_llm_prompt(summary, user_description)

    # Get recommendations from the LLM
    tips = query_llm(prompt)

    # Display the recommendations
    print("\nTips for Improving Data Quality:\n")
    print(tips)


# Example usage
if __name__ == "__main__":
    # Path to your dataset and a description of known issues
    dataset_path = "/Users/derekrosenzweig/Documents/GitHub/build-hours/8-evals/data/draft_stanbot_outputs_n=18.csv"
    user_issues = "The dataset has a lot of missing values in the 'age' and 'income' columns and some outliers in the 'income' field."

    main(dataset_path, user_issues)
