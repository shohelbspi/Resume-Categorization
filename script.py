import os
import shutil
from PyPDF2 import PdfReader
import pickle
import csv

def organize_pdfs(pdf_folder):
    # Load the trained machine learning model and vectorizer using pickle
    with open('models.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
        model = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)

    # Get a list of all PDF files in the input folder
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

    # Initialize a list to store label and file name pairs
    label_file_pairs = []

    # Predict and organize PDFs into folders based on labels
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)

        pdf = PdfReader(pdf_path)

        # Combine text from all pages into a single string
        text = " ".join([page.extract_text() for page in pdf.pages])

        # Transform the extracted text into features using the existing vocabulary
        X_pred = vectorizer.transform([text])

        # Make predictions using the loaded model
        predicted_label = model.predict(X_pred)[0]  # Assuming single label prediction

        # Store the label and file name pair
        label_file_pairs.append((predicted_label, pdf_file))

        # Create a folder based on the predicted label if not already created
        label_folder = os.path.join(pdf_folder, str(predicted_label))  # Convert to string
        os.makedirs(label_folder, exist_ok=True)

        # Move the PDF file to the predicted label-specific folder
        new_pdf_path = os.path.join(label_folder, pdf_file)
        shutil.move(pdf_path, new_pdf_path)

    # Create a CSV file path based on the input folder
    output_csv = os.path.join(pdf_folder, 'categorized_resumes.csv')

    # Write the label and file name pairs to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Predicted Label', 'File Name'])
        csv_writer.writerows(label_file_pairs)

    print("PDF files organized into folders based on predicted labels.")
    print(f"CSV file created at: {output_csv}")

if __name__ == "__main__":
    pdf_folder = 'datas/'
    organize_pdfs(pdf_folder)