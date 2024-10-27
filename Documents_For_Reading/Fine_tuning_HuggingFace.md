# Finetuning HuggingFace model
## Step-1 Using HuggingFace model without finetuning

### Bank Complaints Data
```python
# Download the dataset of bank customer complaints from GitHub
!wget https://github.com/venkatareddykonasani/Datasets/raw/master/Bank_Customer_Complaints/complaints_v2.zip
# Unzip the downloaded dataset
!unzip -o complaints_v2.zip
# Load the complaints data into a DataFrame using pandas
complaints_data = pd.read_csv("/content/complaints_v2.csv")
# Display the first few rows of the complaints data to verify it's loaded correctly
complaints_data.head()
```

This code snippet performs the following actions:

1. **Downloading the Dataset**: It uses the `wget` command to download a zip file containing bank customer complaints data from a specified GitHub repository.

2. **Unzipping the Dataset**: The `unzip` command is used to extract the contents of the downloaded zip file, with the `-o` option to overwrite existing files without prompting.

3. **Loading the Dataset into DataFrame**: It utilizes `pandas` (indicated by `pd`), a powerful Python data analysis toolkit, to read the CSV file (`complaints_v2.csv`) located in the `/content` directory. This step assumes that `pandas` has been imported previously in the code (e.g., `import pandas as pd`).

4. **Displaying the Dataset**: Finally, it displays the first few rows of the complaints dataset using the `.head()` method. This is a useful step for a quick check to ensure the data is loaded as expected and to understand its structure.

Before executing this code, ensure that `pandas` is installed in your Python environment and that you're running this in an environment that supports these commands, such as a Jupyter notebook or a Google Colab notebook.

### Use distilbert model without finetunung
```python
# DistilBERT model
from transformers import pipeline  # Import the pipeline function from the transformers library

# Creating a text classification pipeline with the DistilBERT model, specifying it to run on CUDA (GPU)
distilbert_model = pipeline(task="text-classification",
                            model="distilbert-base-uncased",
                            device="cuda",
                            )

# Sampling 1000 random entries from the complaints data for analysis
sample_data = complaints_data.sample(1000, random_state=42)

# Trimming each text entry to the first 350 words to reduce processing time and memory usage
sample_data["text"] = sample_data["text"].apply(lambda x: " ".join(x.split()[:350]))

# Applying the DistilBERT model to predict the category of each complaint in the sample
sample_data["bert_predicted"] = sample_data["text"].apply(lambda x: distilbert_model(x)[0]["label"])

# Extracting the numerical part of the predicted category (assuming the label is something like 'LABEL_0')
sample_data["bert_predicted_num"] = sample_data["bert_predicted"].apply(lambda x: x[-1])

# Converting the extracted numbers into integers for further analysis or comparison
sample_data["bert_predicted_num"] = sample_data["bert_predicted_num"].astype(int)

# Displaying the first few rows of the modified sample data to verify changes
sample_data.head()
```

This snippet does the following:

- **Setting Up DistilBERT for Text Classification**: It creates a pipeline for text classification using the `distilbert-base-uncased` model. This model is designed to understand the context of the text and classify it accordingly. The model runs on CUDA, indicating the use of GPU for computation, which significantly speeds up processing.

- **Preparing Sample Data**: The script selects a random sample of 1000 entries from the complaints dataset to reduce computation time and focus the analysis on a manageable subset.

- **Text Preprocessing**: For each selected entry, it trims the text to the first 350 words. This is likely done to standardize the input size and reduce computation time, as longer texts could significantly slow down processing without substantially adding to the accuracy of the model for this application.

- **Prediction with DistilBERT**: It applies the DistilBERT model to each trimmed text entry to predict its classification label. The predicted label is expected to be in a format like `'LABEL_X'`, where `X` denotes the class number.

- **Extracting and Processing Predicted Labels**: The script extracts the numeric part from the predicted labels, converts these to integers, and stores them in a new column. This step could be useful for analysis, comparison, or visualization purposes later in the process.

- **Verification**: Finally, it displays the first few rows of the sample data to check the results, including the original text (trimmed to 350 words), the predicted labels, and the numeric part of these labels.

Before running this code, ensure that the `transformers` library is installed in your Python environment and that you have access to a CUDA-capable GPU for the `device="cuda"` configuration to work effectively.


### Accuracy of the model without fine-tuning
```python
from sklearn.metrics import confusion_matrix  # Importing the confusion_matrix function from sklearn.metrics

# Calculating the confusion matrix between the actual labels and the predicted labels
cm = confusion_matrix(sample_data["label"], sample_data["bert_predicted_num"])

# Printing the confusion matrix to see how predictions compare to actual labels
print(cm)

# Calculating the accuracy of the predictions
accuracy = cm.diagonal().sum() / cm.sum()
# Printing the overall accuracy of the model on the sample data
print(accuracy)
```

This snippet calculates and prints the confusion matrix for the predictions made by the DistilBERT model on the sampled dataset, followed by the calculation of the model's accuracy. Here's what each part does:

- **Confusion Matrix Calculation**: It computes the confusion matrix by comparing the actual labels of the data (`sample_data["label"]`) with the labels predicted by the model (`sample_data["bert_predicted_num"]`). The confusion matrix is a table that is often used to describe the performance of a classification model on a set of data for which the true values are known. It shows the actual and predicted classifications in a matrix format, where each row represents the instances in an actual class and each column represents instances in a predicted class.
---
## Step-2 Finetuning the HuggingFace model without our data
```python
# Quietly update and install the accelerate package
!pip -q install accelerate -U
# Quietly install the transformers library along with its PyTorch dependencies
!pip -q install transformers[torch]
# Quietly install the Hugging Face datasets library
!pip -q install datasets

# Importing necessary libraries and classes for model training and data handling
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import Trainer, TrainingArguments  # Trainer and TrainingArguments are imported twice, which is redundant and can be combined into a single import statement
from datasets import load_dataset, DatasetDict, ClassLabel, Dataset
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
import torch  # PyTorch library for deep learning applications
```

This code block sets up the environment for fine-tuning a DistilBERT model by installing necessary libraries and importing essential classes and functions. Here's a breakdown:

1. **Installation Commands**:
    - **`accelerate`**: A PyTorch library from Hugging Face that simplifies running training scripts on multi-GPU, TPU, etc. It's updated quietly (without verbose logs) to the latest version.
    - **`transformers[torch]`**: This installs the Hugging Face `transformers` library along with dependencies for PyTorch integration. It's a key library for accessing pre-trained models like DistilBERT and tools for natural language processing tasks.
    - **`datasets`**: Another library from Hugging Face, which provides easy access to a vast collection of datasets. It's useful for loading and processing data for training and evaluation.

2. **Imports**:
    - **DistilBert Classes**: `DistilBertTokenizer` for tokenizing text inputs and `DistilBertForSequenceClassification` for the sequence classification model.
    - **Training Classes**: `Trainer` for orchestrating the training process and `TrainingArguments` for specifying training configurations.
    - **Datasets**: The `datasets` library functions and classes (`load_dataset`, `DatasetDict`, `ClassLabel`, `Dataset`) for managing datasets.
    - **Pandas and Sklearn**: `pandas` for data manipulation, and `train_test_split` from `sklearn.model_selection` for splitting datasets into training and test sets.
    - **PyTorch**: `torch`, a deep learning library that serves as the backend for the `transformers` library.

Ensure that the environment where this code is run supports all these operations, particularly the installations. This setup is typically used in a Jupyter notebook or a similar interactive Python environment.

```python
#The target variable must be named as "label" - Verify it, before proceeding
print(sample_data.columns)

# Convert the pandas DataFrame `sample_data` to a Hugging Face `Dataset`
Sample_data = Dataset.from_pandas(sample_data)

# Split the dataset into training and testing sets
# Using an 80-20 split for training and testing, respectively
train_test_split = Sample_data.train_test_split(test_size=0.2)  # 80% training, 20% testing

# Creating a DatasetDict to organize the train and test splits
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Displaying the dataset to verify its structure
dataset
```

This code snippet does the following:

1. **Converts the pandas DataFrame to a Hugging Face Dataset**: The `Dataset.from_pandas` method is used to convert `sample_data`, a pandas DataFrame containing the sampled data, into a format (`Dataset`) that can be directly used with Hugging Face's `transformers` and `datasets` libraries.

2. **Splits the Dataset**: It then splits this dataset into training and testing sets using the `train_test_split` method, specifying that 20% of the data should be used for testing (and thus implicitly, 80% for training).

3. **Creates a DatasetDict**: This step organizes the train and test datasets into a `DatasetDict`, which is a convenient way to handle different subsets of a dataset (like training and testing sets) under a single variable. This `DatasetDict` is particularly useful when working with the `Trainer` class from the `transformers` library for training and evaluation.

4. **Displays the DatasetDict**: Finally, it outputs the `dataset` variable to verify its structure, showing the organization of the training and testing sets within the `DatasetDict`.

This approach is essential for preparing the data for model training and evaluation, ensuring that the data is in the correct format and properly partitioned.

### Load the tokenizer
```python
# Load the tokenizer for the 'distilbert-base-uncased' model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Setting up padding token to be the same as the EOS (end of sequence) token
# This aligns padding behavior with DistilBERT's expected inputs
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Adding a special pad token ('[PAD]') to the tokenizer
# This step might not be necessary since pad_token is already set to eos_token, but it explicitly ensures '[PAD]' is recognized as a special token for padding
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Define a function to tokenize the input texts
def tokenize_function(examples):
    # Tokenizes the examples text, ensures they are padded to a maximum length of 512 tokens, and truncated if longer
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Apply the tokenize function to all examples in the dataset
# The 'batched=True' option processes the texts in batches, making this operation faster
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

This code snippet covers the following key steps in preparing the dataset for training with the DistilBERT model:

1. **Loading the Tokenizer**: It initializes the `DistilBertTokenizer` for the 'distilbert-base-uncased' model. Tokenizers convert text inputs into formats that models can understand, including splitting texts into tokens, converting tokens to IDs, and managing special tokens for beginning, padding, and ending sequences.

2. **Padding Setup**: The padding token is set to be the same as the end-of-sequence (EOS) token. This configuration ensures that padding behavior aligns with what the DistilBERT model expects for its inputs. It also adds a special pad token '[PAD]' to the tokenizer, although setting the pad token to the EOS token might already cover the need for padding.

3. **Tokenization Function**: Defines a function to tokenize the text data. This function applies the tokenizer to the "text" field of the input examples, ensuring that each example is padded to a maximum length of 512 tokens or truncated if it exceeds this length. These settings are crucial for processing texts consistently and fitting them into the model's input size requirements.

4. **Applying Tokenization**: The tokenization function is applied to all examples in the dataset using the `map` method with `batched=True`, which efficiently processes the data in batches.

This approach ensures that the dataset is appropriately prepared and tokenized, fitting the DistilBERT model's requirements for training or evaluation. The `tokenized_datasets` variable now contains the input IDs and attention masks necessary for the model to process the text data.

### Load and Train the model
```python
# Initialize the DistilBERT model for sequence classification with the 'distilbert-base-uncased' pre-trained model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                            num_labels=2,  # Adjust this based on the number of classes in your dataset
                                                            pad_token_id=tokenizer.eos_token_id)  # Set the padding token ID to align with the tokenizer's EOS token ID

# Set up the training arguments specifying the directory for saving results, the number of training epochs, and the logging directory
training_args = TrainingArguments(
    output_dir="./results_bert_custom",  # Directory where the training results will be saved
    num_train_epochs=1,  # Number of epochs to train for
    logging_dir="./logs_bert_custom",  # Directory where the training logs will be saved
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
)

# Initialize the Trainer with the model, training arguments, and the train and evaluation datasets
trainer = Trainer(
    model=model,  # The pre-initialized DistilBERT model
    args=training_args,  # Training arguments specifying the training setup
    train_dataset=tokenized_datasets['train'],  # The tokenized training dataset
    eval_dataset=tokenized_datasets['test'],  # The tokenized evaluation dataset
)

# Start the training process
trainer.train()
```

This code snippet sets up and initiates the training process for a DistilBERT model tailored for sequence classification. The process involves several key steps:

1. **Model Initialization**: A `DistilBertForSequenceClassification` model is initialized with the pre-trained 'distilbert-base-uncased' configuration. The `num_labels` parameter is set according to the number of classes you have (in this case, 2), and the padding token ID is aligned with the tokenizer's end-of-sequence (EOS) token ID to ensure consistent handling of padding.

2. **Training Arguments Setup**: `TrainingArguments` are configured to define various aspects of the training process, including the output directory for saving results, the number of training epochs, the logging directory for storing logs, and the evaluation strategy (here, evaluating at the end of each epoch).

3. **Trainer Initialization**: A `Trainer` object is instantiated with the model, the training arguments, and the tokenized training and evaluation datasets. The `Trainer` class is a flexible tool for training and evaluating transformers models, handling the training loop internally and abstracting away many of the details involved in training transformers models.

4. **Training Start**: The training process is started by calling the `train()` method on the `Trainer` object. This will train the model on the specified training dataset for the given number of epochs, evaluating its performance on the evaluation dataset at the end of each epoch according to the defined evaluation strategy.

Before running this code, ensure that your environment has enough resources (CPU/GPU) to support model training, and adjust the `num_labels` parameter to match the actual number of classes in your dataset.

### Save the model
```python
# Define the directory where you want to save your model and tokenizer
model_dir = "./distilbert_finetuned"

# Save the model to the specified directory
model.save_pretrained(model_dir)

# Save the tokenizer to the same directory
tokenizer.save_pretrained(model_dir)

# Define a function for making predictions with the finetuned model
def make_prediction(text):
    # Prepare the input text for the model
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Move the inputs to the GPU
    inputs = inputs.to(torch.device("cuda:0"))
    # Get model predictions
    outputs = model(**inputs)
    # Extract the predicted class (the one with the highest score) from the logits
    predictions = outputs.logits.argmax(-1)
    # Move the predictions back to CPU and convert to numpy for easy handling
    predictions = predictions.detach().cpu().numpy()
    # Return the predicted class
    return predictions

# Apply the prediction function to the 'text' column of the sample_data DataFrame
# Note: If running this line results in performance issues or out-of-memory errors, consider applying predictions in batches.
sample_data["finetuned_predicted"] = sample_data["text"].apply(lambda x: make_prediction(str(x))[0])
```

This snippet includes steps for saving a finetuned DistilBERT model and its tokenizer, as well as defining and using a function to make predictions on new text data using the finetuned model. Key points include:

- **Saving the Model and Tokenizer**: Both the finetuned model and its associated tokenizer are saved to a specified directory. This is useful for deployment or future use without needing to retrain the model.

- **Prediction Function**: `make_prediction` is a function that takes a piece of text as input, tokenizes it, sends the tokenized input to the GPU, gets predictions from the model, and processes these predictions to return the most likely class for each input text.

- **Applying the Prediction Function**: The `make_prediction` function is applied to each text entry in the `sample_data` DataFrame, with the predictions stored in a new column named `finetuned_predicted`. This demonstrates how the finetuned model can be used for inference on unseen data.

Please note, the `inputs = inputs.to(torch.device("cuda:0"))` line assumes you have a CUDA-enabled GPU available. If you're running this code on a CPU, you should modify this line to `inputs = inputs.to(torch.device("cpu"))` or remove it entirely, as PyTorch defaults to using the CPU when no device is specified. Additionally, if processing a large amount of data, consider batch processing to improve efficiency and manage resource use.

```python
from sklearn.metrics import confusion_matrix
# Create the confusion matrix
cm1 = confusion_matrix(sample_data["label"], sample_data["finetuned_predicted"])
print(cm1)
accuracy1=cm1.diagonal().sum()/cm1.sum()
print(accuracy1)
```

### Loading a Pre-trained model 

```python
#Code to donwload the distilbert model
!gdown --id 1785J3ir19RaZP3ebbFvWUX88PMaBouro -O distilbert_finetuned_V1.zip
!unzip -o -j distilbert_finetuned_V1.zip -d distilbert_finetuned_V1

model_v1 = DistilBertForSequenceClassification.from_pretrained('/content/distilbert_finetuned_V1')
model_v1.to("cuda:0")

def make_prediction(text):
  new_complaint=text
  inputs=tokenizer(new_complaint, return_tensors="pt")
  inputs = inputs.to(torch.device("cuda:0"))
  outputs=model_v1(**inputs)
  predictions=outputs.logits.argmax(-1)
  predictions=predictions.detach().cpu().numpy()
  return(predictions)

sample_data_large=complaints_data.sample(n=10000, random_state=55)
sample_data_large["finetuned_predicted"]=sample_data_large["text"].apply(lambda x: make_prediction(str(x)[:350])[0])

from sklearn.metrics import confusion_matrix
# Create the confusion matrix
cm1 = confusion_matrix(sample_data_large["label"], sample_data_large["finetuned_predicted"])
print(cm1)
accuracy1=cm1.diagonal().sum()/cm1.sum()
print(accuracy1)

```

This code snippet demonstrates the process of downloading a finetuned DistilBERT model, loading it for inference, making predictions on a larger sample of data, and evaluating the model's performance through a confusion matrix and accuracy calculation. Here's a breakdown of the steps:

### Downloading and Unzipping the Model
- It uses `gdown` to download a zip file containing the finetuned DistilBERT model from Google Drive.
- The zip file is then extracted into a specified directory.

### Loading the Finetuned Model
- Loads the finetuned DistilBERT model from the extracted directory.
- Moves the model to the GPU to enable fast inference.

### Defining the Prediction Function
- The `make_prediction` function tokenizes input text, processes it through the model to get predictions, and returns the predicted class for the input text.

### Making Predictions on a Larger Dataset
- Samples a larger portion of the original complaints data for prediction.
- Truncates texts to the first 350 tokens before making predictions, assuming longer texts may not significantly affect prediction outcomes.
- Applies the `make_prediction` function to each text in the sampled data, storing predictions in a new column.

### Evaluating Model Performance
- Constructs a confusion matrix comparing actual labels with predicted labels.
- Calculates the accuracy of the model as the proportion of correctly predicted instances.

### Important Considerations:
- Ensure the GPU is available (`cuda:0`) for model inference; otherwise, adjust the device accordingly.
- When applying the `make_prediction` function to a large dataset, consider the computational cost. The current approach processes texts one by one, which might be slower for large datasets. Batch processing can be more efficient.
- Truncating text to 350 tokens before prediction is a practical choice to manage computational load and memory usage, but the optimal length may vary based on the dataset and the specific requirements of the task.
- The accuracy calculation is straightforward, providing a quick measure of model performance. However, for a detailed evaluation, especially with imbalanced datasets, consider additional metrics such as precision, recall, and F1-score.

This approach demonstrates a practical workflow for applying a finetuned NLP model to a real-world task, from downloading and loading the model to making predictions and evaluating performance.
