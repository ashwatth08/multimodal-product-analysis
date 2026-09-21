# multimodal-product-analysis
Multimodal Product Analysis Using Image Classification and Customer Review Sentiment Analysis

# Multimodal Product Analysis Using Image Classification and Customer Review Sentiment Analysis

## Project Overview

This mini project presents a multimodal product analysis system that combines product image classification with customer review sentiment analysis.

The system analyzes both the visual condition of a product and the sentiment expressed in the customer review. By comparing these two modalities, the system attempts to identify whether the review is consistent with the actual condition of the product.

## Objectives

* Analyze product images using deep learning.
* Analyze customer reviews using sentiment analysis.
* Combine image and text information.
* Detect possible mismatches between product condition and customer review.
* Identify potentially false or inconsistent reviews.

## System Workflow

1. Input a product image.
2. Input the corresponding customer review.
3. Process the product image using the image classification model.
4. Analyze the customer review using sentiment analysis.
5. Combine the image and text predictions.
6. Determine whether the review is consistent with the product condition.
7. Display the final prediction.

## Technologies Used

* Python
* PyTorch
* Computer Vision
* Natural Language Processing
* Sentiment Analysis
* Deep Learning
* Streamlit
* Pandas
* NumPy
* Scikit-learn

## Project Files

```text
Mini-Project/
│
├── app.py
├── train.py
├── prepare_data.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── MVSA_Multiple/
├── MVSA_Single/
│
├── data/
│   ├── clean_mvsa_paired.csv
│   └── dataset.csv
│
└── models/
```

## Main Components

### `app.py`

Runs the application and provides the user interface for submitting product images and customer reviews.

### `train.py`

Contains the model training process.

### `prepare_data.py`

Performs dataset preparation and preprocessing.

### `MVSA_Multiple`

Contains resources related to the MVSA Multiple multimodal sentiment dataset.

### `MVSA_Single`

Contains resources related to the MVSA Single multimodal sentiment dataset.

## Example Predictions

### Case 1: Good Product + Negative Review

If the product image indicates that the product is in good condition but the customer review describes the product negatively, the system can identify a possible review-product mismatch.

### Case 2: Defective Product + Positive Review

If the image indicates that the product is damaged or defective while the review describes the product positively, the system can identify another possible mismatch.

### Case 3: Consistent Product and Review

If the product condition and customer review indicate the same overall condition, the system considers the two modalities consistent.

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Move into the project directory:

```bash
cd Mini-Project
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Model

The trained multimodal model is used to combine information obtained from the image and customer review.

If the trained model file is not included in the repository because of its size, it can be downloaded separately and placed in the required model directory.

## Future Enhancements

* Improve image classification accuracy.
* Support additional product categories.
* Improve multilingual review analysis.
* Add explainable AI for image predictions.
* Detect more types of review-image inconsistencies.
* Deploy the system as a web or mobile application.
* Integrate real-time e-commerce product review analysis.
