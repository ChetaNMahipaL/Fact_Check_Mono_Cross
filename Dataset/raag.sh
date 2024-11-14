#!/bin/bash

# Check for required arguments: OpenAI API Key and dataset path
if [ "$#" -ne 2 ]; then
    echo "Usage: ./fine_tune_gpt.sh <OPENAI_API_KEY> <DATASET_PATH>"
    exit 1
fi

# Set variables from arguments
OPENAI_API_KEY=$1
DATASET_PATH=$2
JSONL_FILE="formatted_data.jsonl"

# Ensure the OpenAI CLI is configured
echo "Setting up OpenAI API Key..."
export OPENAI_API_KEY=$OPENAI_API_KEY

# Step 1: Format Dataset into JSONL
echo "Formatting dataset into JSONL format..."
# {"prompt": "Input: 'The sentence to verify'\nFact Check: 'The fact check claim'\nOutput:", "completion": " 1"}


# Initialize the JSONL file
> $JSONL_FILE

# Read CSV dataset line by line and convert to JSONL
while IFS=, read -r sentence fact_check label; do
    # Skip header row or empty lines
    if [[ "$sentence" == "sentence" || -z "$sentence" ]]; then
        continue
    fi

    # Format prompt and completion
    prompt="Input: \"$sentence\"\nFact Check: \"$fact_check\"\nOutput:"
    completion=" $label"
    
    # Write to JSONL file
    echo "{\"prompt\": \"$prompt\", \"completion\": \"$completion\"}" >> $JSONL_FILE
done < "$DATASET_PATH"

echo "Data formatted into JSONL at $JSONL_FILE."

# Step 2: Upload and Fine-Tune
echo "Starting fine-tuning job with OpenAI..."

# Run fine-tuning command
FINE_TUNE_ID=$(openai api fine_tunes.create -t "$JSONL_FILE" -m "gpt-3.5-turbo" | jq -r '.id')

# Check if fine-tune job started successfully
if [[ -z "$FINE_TUNE_ID" ]]; then
    echo "Failed to start fine-tuning job. Please check your data and API key."
    exit 1
fi

echo "Fine-tuning job started successfully with ID: $FINE_TUNE_ID"
echo "You can track the job status with: openai api fine_tunes.get -i $FINE_TUNE_ID"

# Optional: Wait for job completion
echo "Waiting for fine-tuning to complete..."

while true; do
    STATUS=$(openai api fine_tunes.get -i $FINE_TUNE_ID | jq -r '.status')
    echo "Current status: $STATUS"
    
    if [[ "$STATUS" == "succeeded" ]]; then
        MODEL_NAME=$(openai api fine_tunes.get -i $FINE_TUNE_ID | jq -r '.fine_tuned_model')
        echo "Fine-tuning completed successfully. Model name: $MODEL_NAME"
        break
    elif [[ "$STATUS" == "failed" ]]; then
        echo "Fine-tuning job failed. Please review your data and try again."
        exit 1
    else
        # Wait for a few seconds before checking status again
        sleep 60
    fi
done

echo "Fine-tuning process completed. Your model is ready to use as: $MODEL_NAME"
echo "Use this model by passing '$MODEL_NAME' as the model name in your API calls."
